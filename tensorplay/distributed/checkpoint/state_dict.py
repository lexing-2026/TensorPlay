from __future__ import annotations

import contextlib
import copy
import functools
import gc
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable, Generator
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, cast

import tensorplay as tp

from tensorplay.nn.modules.module import Module
from tensorplay.optim.optimizer import Optimizer

try:
    from tensorplay.nn.modules.module import _IncompatibleKeys
except ImportError:
    _IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])

try:
    from tensorplay.distributed._shard.sharded_tensor.api import ShardedTensor
except ImportError:
    ShardedTensor = ()

__all__ = [
    "FQNS_T",
    "PrimitiveType",
    "ValueType",
    "DictValueType",
    "ListDictValueType",
    "OptimizerStateType",
    "StateDictOptions",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "get_state_dict",
    "set_model_state_dict",
    "set_optimizer_state_dict",
    "set_state_dict",
]

_FLAT_PARAM = "_flat_param"
_PG = "param_groups"
_PARAMS = "params"
_STATE = "state"
_EXTRA_STATE_NAME = "_extra_state"
_patched_state_dict: set[Callable[..., Any]] = set()

FQNS_T = set[str]
PrimitiveType = Any
ValueType = Any
DictValueType = dict[str, Any]
ListDictValueType = list[dict[str, Any]]
OptimizerStateType = dict[str, Any]


@contextlib.contextmanager
def _gc_context() -> Generator[None, None, None]:
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@dataclass
class StateDictOptions:
    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True
    broadcast_from_rank0: bool = False
    flatten_optimizer_state_dict: bool = False
    dsd_fqn_modifiers: str = "_fqn_modifiers"


@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: dict[Any, Any] = field(default_factory=dict)
    shared_params_mapping: dict[Any, Any] = field(default_factory=dict)
    submodule_prefixes: set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable[..., Any] = contextlib.nullcontext
    fsdp_modules: list[Any] = field(default_factory=list)


class _EXTRA_STATE:
    pass


def _is_sharded(value: Any) -> bool:
    return bool(ShardedTensor) and isinstance(value, ShardedTensor)


def _is_distributed(value: Any) -> bool:
    return hasattr(value, "device_mesh") and callable(getattr(value, "to_local", None))


def _is_tensor_value(value: Any) -> bool:
    return isinstance(value, tp.Tensor) or _is_distributed(value) or _is_sharded(value)


def _clone_value(value: Any, memo: dict[int, Any] | None = None) -> Any:
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if _is_distributed(value):
        clone = value.detach().clone()
        memo[id(value)] = clone
        return clone
    if _is_sharded(value):
        metadata = copy.copy(value.metadata())
        if hasattr(metadata, "shards_metadata"):
            copied_metadata = []
            for item in metadata.shards_metadata:
                item_copy = copy.copy(item)
                if hasattr(item, "shard_offsets"):
                    object.__setattr__(
                        item_copy, "shard_offsets", list(item.shard_offsets)
                    )
                if hasattr(item, "shard_sizes"):
                    object.__setattr__(
                        item_copy, "shard_sizes", list(item.shard_sizes)
                    )
                copied_metadata.append(item_copy)
            metadata.shards_metadata = copied_metadata
        if hasattr(metadata, "tensor_properties"):
            metadata.tensor_properties = copy.copy(metadata.tensor_properties)
        shards = []
        for shard in value.local_shards():
            item = copy.copy(shard.metadata)
            if hasattr(shard.metadata, "shard_offsets"):
                object.__setattr__(item, "shard_offsets", list(shard.metadata.shard_offsets))
            if hasattr(shard.metadata, "shard_sizes"):
                object.__setattr__(item, "shard_sizes", list(shard.metadata.shard_sizes))
            shards.append(type(shard)(_clone_value(shard.tensor, memo), item))
        clone = type(value)._init_from_local_shards_and_global_metadata(
            shards,
            metadata,
            getattr(value, "_sharding_spec", None),
            getattr(value, "_process_group", None),
        )
        memo[id(value)] = clone
        return clone
    if isinstance(value, tp.Tensor):
        clone = value.detach().clone()
        memo[id(value)] = clone
        for name, attribute in getattr(value, "__dict__", {}).items():
            try:
                setattr(clone, name, _clone_value(attribute, memo))
            except (AttributeError, TypeError):
                continue
        return clone
    if isinstance(value, dict):
        clone = {}
        memo[id(value)] = clone
        clone.update((key, _clone_value(child, memo)) for key, child in value.items())
        return clone
    if isinstance(value, list):
        clone = [_clone_value(child, memo) for child in value]
        memo[id(value)] = clone
        return clone
    if isinstance(value, tuple):
        clone = tuple(_clone_value(child, memo) for child in value)
        memo[id(value)] = clone
        return clone
    return copy.deepcopy(value, memo)


def _unwrap(model: Any) -> Any:
    current = model
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        current = current.module
    return current


def _get_fqns(
    model: Any,
    name: str,
    dsd_fqn_modifiers: str = "_fqn_modifiers",
    skip_ddp_prefix: bool = True,
    skip_compiler_prefix: bool = True,
) -> FQNS_T:
    del skip_compiler_prefix
    name = name.replace("_checkpoint_wrapper.", "")
    parts = name.split(".") if name else []
    current = model
    result: list[str] = []
    for part in parts:
        if skip_ddp_prefix and part == "module" and hasattr(current, "module"):
            current = current.module
            continue
        if part == "_orig_mod" and hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        if part == _FLAT_PARAM:
            flat_param = getattr(current, _FLAT_PARAM, None)
            if flat_param is None:
                state = getattr(current, "_fsdp_state", None)
                flat_param = getattr(state, _FLAT_PARAM, None)
            fqns = getattr(flat_param, "_fqns", None)
            if fqns:
                prefix = ".".join(result)
                return {
                    f"{prefix}.{fqn}" if prefix else str(fqn)
                    for fqn in fqns
                }
        modifiers = getattr(current, dsd_fqn_modifiers, None)
        if callable(modifiers):
            removed = modifiers().get(part)
            if removed is not None and hasattr(current, removed):
                part = removed
        result.append(part)
        if part != _EXTRA_STATE_NAME:
            current = getattr(current, part, current)
    return {".".join(result)}


def _iterate_valid_model_state(model: Any, dsd_fqn_modifiers: str = "_fqn_modifiers") -> Generator[tuple[str, Any], None, None]:
    visited: set[int] = set()

    def recurse(module: Any, prefix: str) -> Generator[tuple[str, Any], None, None]:
        if id(module) in visited:
            return
        visited.add(id(module))
        base = f"{prefix}." if prefix else ""
        named_children = getattr(module, "named_children", lambda: ())
        for name, child in named_children():
            modifiers = getattr(module, dsd_fqn_modifiers, None)
            child_name = name
            if callable(modifiers):
                removed = modifiers().get(name)
                if removed is not None:
                    child_name = removed
            yield from recurse(child, f"{prefix}.{child_name}" if prefix else child_name)
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        named_buffers = getattr(module, "named_buffers", lambda **_: ())
        named_parameters = getattr(module, "named_parameters", lambda **_: ())
        for name, value in chain(named_buffers(recurse=False), named_parameters(recurse=False)):
            if name in non_persistent:
                continue
            yield f"{base}{name}", value
        extra = getattr(module.__class__, "get_extra_state", None)
        base_extra = getattr(Module, "get_extra_state", None)
        if extra is not None and extra is not base_extra:
            yield f"{base}{_EXTRA_STATE_NAME}", _EXTRA_STATE()

    yield from recurse(model, "")


def _param_key(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return id(value)


def _param_fqns(info: _StateDictInfo, value: Any) -> set[str]:
    result = info.fqn_param_mapping.get(_param_key(value))
    if result is None:
        result = info.fqn_param_mapping.get(id(value), set())
    return set(result) if isinstance(result, (set, list, tuple)) else set()


def _verify_options(
    model: Any,
    optims: tuple[Any, ...],
    optim_only: bool,
    *,
    submodules: set[Any] | None = None,
    options: StateDictOptions | None = None,
) -> _StateDictInfo:
    if optim_only and not optims:
        raise RuntimeError("optimizers are required when optim_only is enabled")
    options = options or StateDictOptions()
    fqn_param_mapping: dict[Any, Any] = {}
    shared_params_mapping: dict[Any, Any] = {}
    for name, value in _iterate_valid_model_state(model, options.dsd_fqn_modifiers):
        if isinstance(value, _EXTRA_STATE):
            continue
        fqns = _get_fqns(model, name, options.dsd_fqn_modifiers)
        key = _param_key(value)
        previous = fqn_param_mapping.get(key)
        if previous is None:
            fqn_param_mapping[key] = set(fqns)
        else:
            previous.update(fqns)
            shared_params_mapping[key] = previous
        for fqn in fqns:
            fqn_param_mapping[fqn] = value
    prefixes: set[str] = set()
    if submodules:
        for name, module in getattr(model, "named_modules", lambda: ())():
            if module in submodules:
                fqn = next(iter(_get_fqns(model, name)), "")
                prefixes.add(f"{fqn}." if fqn else "")
    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError("full_state_dict must be enabled for broadcast_from_rank0")
    fsdp_modules: list[Any] = []
    fsdp_context: Callable[..., Any] = contextlib.nullcontext
    try:
        from tensorplay.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        fsdp_modules = FullyShardedDataParallel.fsdp_modules(model)
        if fsdp_modules:
            if options.full_state_dict:
                state_type = StateDictType.FULL_STATE_DICT
                state_config = FullStateDictConfig(
                    offload_to_cpu=options.cpu_offload,
                    rank0_only=options.cpu_offload,
                )
                optim_config = FullOptimStateDictConfig(
                    offload_to_cpu=options.cpu_offload,
                    rank0_only=options.cpu_offload or options.broadcast_from_rank0,
                )
            else:
                state_type = StateDictType.SHARDED_STATE_DICT
                state_config = ShardedStateDictConfig(
                    offload_to_cpu=options.cpu_offload,
                )
                optim_config = ShardedOptimStateDictConfig(
                    offload_to_cpu=options.cpu_offload,
                )
            fsdp_context = functools.partial(
                FullyShardedDataParallel.state_dict_type,
                model,
                state_type,
                state_config,
                optim_config,
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        fsdp_modules = []
        fsdp_context = contextlib.nullcontext
    return _StateDictInfo(
        **asdict(options),
        fqn_param_mapping=fqn_param_mapping,
        shared_params_mapping=shared_params_mapping,
        submodule_prefixes=prefixes,
        handle_model=not optim_only,
        handle_optim=bool(optims),
        fsdp_context=fsdp_context,
        fsdp_modules=fsdp_modules,
    )


def _verify_state_dict(
    model_state_dict: dict[str, Any],
    optim_state_dict: dict[str, Any],
    info: _StateDictInfo,
) -> None:
    if info.handle_model and not model_state_dict and info.strict and not info.broadcast_from_rank0:
        raise RuntimeError("model state dictionary is empty")
    if info.handle_optim and not optim_state_dict and info.strict and not info.broadcast_from_rank0:
        raise RuntimeError("optimizer state dictionary is empty")
    for key in model_state_dict:
        if _FLAT_PARAM in key:
            raise RuntimeError(f"invalid model state key {key}")


def _state_dict_fn(obj: Any, api: str) -> Callable[..., Any]:
    call = getattr(obj, api)
    if call in _patched_state_dict:
        return functools.partial(getattr(obj.__class__, api), obj)
    return call


def _get_fsdp_process_group(model: Any, info: _StateDictInfo) -> Any:
    if not info.fsdp_modules:
        return None
    candidate = info.fsdp_modules[0]
    if hasattr(model, "process_group"):
        candidate = model
    process_group = getattr(candidate, "process_group", None)
    if isinstance(process_group, tuple):
        return None
    if process_group is not None:
        return process_group
    state = getattr(candidate, "_fsdp_state", None)
    return getattr(state, "process_group", None)


def _offload_value(value: Any) -> Any:
    if _is_distributed(value):
        local = value.to_local().to(device="cpu")
        return value.__class__(local, value.device_mesh, value.placements, shape=value.shape)
    if _is_sharded(value):
        return value.cpu() if callable(getattr(value, "cpu", None)) else value
    if isinstance(value, tp.Tensor):
        return value.to(device="cpu")
    if isinstance(value, dict):
        return {key: _offload_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_offload_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_offload_value(child) for child in value)
    return value


def _maybe_full_or_cpu_state_dict(state_dict: dict[str, Any], info: _StateDictInfo) -> dict[str, Any]:
    if info.full_state_dict:
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            if _is_distributed(value) and callable(getattr(value, "gather", None)):
                value = value.gather()
            elif _is_sharded(value) and callable(getattr(value, "gather", None)):
                value = value.gather()
            result[key] = value
        state_dict = result
    if info.cpu_offload:
        state_dict = _offload_value(state_dict)
    return state_dict


def _get_model_state_dict(model: Any, info: _StateDictInfo) -> dict[str, Any]:
    if not info.handle_model:
        return {}
    source = model if info.fsdp_modules else _unwrap(model)
    with info.fsdp_context():
        state = _state_dict_fn(source, "state_dict")()
    result: dict[str, Any] = {}
    parameter_map = dict(getattr(model, "named_parameters", lambda: ())())
    for key, value in state.items():
        fqn = next(iter(_get_fqns(model, key, info.dsd_fqn_modifiers)), key)
        if info.submodule_prefixes and not any(fqn.startswith(prefix) for prefix in info.submodule_prefixes):
            continue
        if info.ignore_frozen_params:
            parameter = parameter_map.get(key)
            if parameter is not None and not bool(parameter.requires_grad):
                continue
        result[fqn] = _clone_value(value)
    return _maybe_full_or_cpu_state_dict(result, info)


def _load_model_state_dict(model: Any, state_dict: dict[str, Any], info: _StateDictInfo) -> Any:
    if not info.handle_model or not state_dict:
        return _IncompatibleKeys([], [])
    source = model if info.fsdp_modules else _unwrap(model)
    with info.fsdp_context():
        live = _state_dict_fn(source, "state_dict")()
    actual: dict[str, Any] = {}
    live_fqns: set[str] = set()
    for key in live:
        fqn = next(iter(_get_fqns(model, key, info.dsd_fqn_modifiers)), key)
        live_fqns.add(fqn)
        if fqn in state_dict:
            actual[key] = state_dict[fqn]
    if info.strict:
        missing = [key for key in state_dict if key not in live_fqns]
        if missing:
            raise RuntimeError(f"missing model keys: {missing}")
    try:
        with info.fsdp_context():
            return _state_dict_fn(source, "load_state_dict")(actual, strict=info.strict)
    except AttributeError:
        missing: list[str] = []
        unexpected = [key for key in state_dict if key not in live_fqns]
        for key, value in actual.items():
            target = live[key]
            if not _is_tensor_value(target) or not _is_tensor_value(value):
                missing.append(key)
                continue
            target.copy_(value.to(device=target.device))
        if info.strict and missing:
            raise RuntimeError(f"model keys could not be loaded: {missing}")
        return _IncompatibleKeys(missing, unexpected)


def _init_optim_state(optim: Any) -> None:
    if getattr(optim, "state", None):
        return
    changed: list[tuple[dict[str, Any], Any]] = []
    for group in optim.param_groups:
        for param in group[_PARAMS]:
            if getattr(param, "grad", None) is None and getattr(param, "requires_grad", False):
                param.grad = tp.zeros_like(param)
        if "lr" in group:
            changed.append((group, group["lr"]))
            group["lr"] = 0.0
    try:
        if changed:
            optim.step()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    finally:
        for group, value in changed:
            group["lr"] = value
        zero_grad = getattr(optim, "zero_grad", None)
        if callable(zero_grad):
            zero_grad(set_to_none=True)


def _name_by_param(model: Any) -> dict[Any, str]:
    return {
        _param_key(param): next(iter(_get_fqns(model, name)), name)
        for name, param in getattr(model, "named_parameters", lambda: ())()
    }


def _get_optim_state_dict(model: Any, optimizers: tuple[Any, ...], info: _StateDictInfo) -> OptimizerStateType:
    if not info.handle_optim:
        return {}
    result: OptimizerStateType = {_STATE: {}, _PG: []}
    names = _name_by_param(model)
    for optim in optimizers:
        _init_optim_state(optim)
        osd = _state_dict_fn(optim, "state_dict")()
        if info.fsdp_modules:
            from tensorplay.distributed.fsdp import FullyShardedDataParallel

            with info.fsdp_context():
                osd = FullyShardedDataParallel.optim_state_dict(
                    model,
                    optim,
                    osd,
                    group=_get_fsdp_process_group(model, info),
                )
            if not osd:
                continue
            for key in list(osd.get(_STATE, {})):
                if "_orig_mod." in str(key):
                    osd[_STATE][str(key).replace("_orig_mod.", "")] = osd[_STATE].pop(key)
            for group in osd.get(_PG, []):
                group[_PARAMS] = [str(key).replace("_orig_mod.", "") for key in group[_PARAMS]]
            result[_PG].extend(
                {
                    key: _clone_value(value)
                    for key, value in group.items()
                }
                for group in osd.get(_PG, [])
            )
            result[_STATE].update(
                {
                    key: _clone_value(value)
                    for key, value in osd.get(_STATE, {}).items()
                }
            )
            continue
        id_to_param: dict[Any, Any] = {}
        for group, saved_group in zip(optim.param_groups, osd.get(_PG, ())):
            for param, param_id in zip(group[_PARAMS], saved_group[_PARAMS]):
                id_to_param[param_id] = param
            saved_params = [
                names.get(_param_key(param), param_id)
                for param, param_id in zip(group[_PARAMS], saved_group[_PARAMS])
            ]
            result[_PG].append(
                {
                    key: _clone_value(value) if key != _PARAMS else saved_params
                    for key, value in saved_group.items()
                }
            )
        for param_id, value in osd.get(_STATE, {}).items():
            parameter = id_to_param.get(param_id)
            fqn = names.get(_param_key(parameter), param_id)
            result[_STATE][fqn] = _clone_value(value)
    if info.flatten_optimizer_state_dict:
        return _flatten_optim_state_dict(result)
    return cast(OptimizerStateType, _maybe_full_or_cpu_state_dict(result, info))


def _flatten_optim_state_dict(state_dict: OptimizerStateType) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        else:
            flattened[prefix] = value

    for fqn, value in state_dict.get(_STATE, {}).items():
        visit(value, f"{_STATE}.{fqn}")
    for group in state_dict.get(_PG, []):
        for fqn in group.get(_PARAMS, []):
            for key, value in group.items():
                if key != _PARAMS:
                    flattened[f"{_PG}.{fqn}.{key}"] = value
    return flattened


def _unflatten_optim_state_dict(optim: Any, state_dict: dict[str, Any], info: _StateDictInfo) -> OptimizerStateType:
    def reconstruct(prefix: str) -> Any:
        direct = state_dict.get(prefix)
        if direct is not None or prefix in state_dict:
            return direct
        nested: dict[str, Any] = {}
        marker = f"{prefix}."
        for key, value in state_dict.items():
            if not key.startswith(marker):
                continue
            remaining = key[len(marker) :]
            parts = remaining.split(".")
            current = nested
            for part in parts[:-1]:
                child = current.get(part)
                if child is None:
                    child = {}
                    current[part] = child
                if not isinstance(child, dict):
                    raise ValueError(f"optimizer state key collision at {key}")
                current = child
            if parts:
                current[parts[-1]] = value
        return nested

    state: dict[str, Any] = {}
    groups: list[dict[str, Any]] = []
    for param_group in optim.param_groups:
        params: list[str] = []
        for param in param_group[_PARAMS]:
            fqns = _param_fqns(info, param)
            if not fqns:
                fqns = {str(id(param))}
            selected = sorted(fqns)
            if len(selected) > 1:
                selected = [
                    fqn
                    for fqn in selected
                    if any(f"{_PG}.{fqn}." in key for key in state_dict)
                ] or selected[:1]
            for fqn in selected:
                params.append(fqn)
                if getattr(param, "requires_grad", False):
                    live_state = getattr(optim, "state", {}).get(param, {})
                    loaded_state: dict[str, Any] = {}
                    for state_name in live_state:
                        loaded_state[state_name] = reconstruct(
                            f"{_STATE}.{fqn}.{state_name}"
                        )
                    if loaded_state:
                        state[fqn] = loaded_state
        group: dict[str, Any] = {_PARAMS: params}
        if params:
            first_fqn = params[0]
            for key in param_group:
                if key == _PARAMS:
                    continue
                prefix = f"{_PG}.{first_fqn}.{key}"
                if prefix in state_dict:
                    group[key] = state_dict[prefix]
        groups.append(group)
    if not groups:
        groups = [{_PARAMS: []} for _ in optim.param_groups]
    return {_STATE: state, _PG: groups}


def _split_optim_state_dict(model: Any, optim: Any, optim_state_dict: OptimizerStateType, info: _StateDictInfo) -> OptimizerStateType:
    if _STATE not in optim_state_dict:
        optim_state_dict = _unflatten_optim_state_dict(optim, optim_state_dict, info)
    result_state: dict[int, Any] = {}
    result_groups: list[dict[str, Any]] = [{_PARAMS: []} for _ in optim.param_groups]
    loaded_groups = optim_state_dict.get(_PG, [])
    loaded_state = optim_state_dict.get(_STATE, {})
    group_for_fqn: dict[str, int] = {}
    for loaded_index, loaded_group in enumerate(loaded_groups):
        for fqn in loaded_group.get(_PARAMS, []):
            group_for_fqn[str(fqn)] = loaded_index

    next_id = 0
    for group_index, group in enumerate(optim.param_groups):
        local_group = result_groups[group_index]
        loaded_values: list[dict[str, Any]] = []
        for param in group[_PARAMS]:
            fqns = sorted(_param_fqns(info, param))
            if not fqns:
                fqns = [_name_by_param(model).get(_param_key(param), str(id(param)))]
            fqn = next(
                (candidate for candidate in fqns if candidate in loaded_state),
                next(
                    (
                        candidate
                        for candidate in fqns
                        if candidate in group_for_fqn
                    ),
                    fqns[0],
                ),
            )
            param_id = next_id
            next_id += 1
            local_group[_PARAMS].append(param_id)
            if fqn in loaded_state:
                result_state[param_id] = _clone_value(loaded_state[fqn])
            source_group_index = group_for_fqn.get(fqn)
            if source_group_index is not None and source_group_index < len(loaded_groups):
                loaded_values.append(loaded_groups[source_group_index])
            elif group_index < len(loaded_groups):
                loaded_values.append(loaded_groups[group_index])
            elif info.strict and getattr(param, "requires_grad", False):
                raise RuntimeError(
                    f"missing optimizer state for parameter '{fqn}'"
                )
        if loaded_values:
            first = loaded_values[0]
            for key, value in first.items():
                if key != _PARAMS:
                    local_group[key] = _clone_value(value)
    return {_STATE: result_state, _PG: result_groups}


def _load_optim_state_dict(model: Any, optimizers: tuple[Any, ...], state_dict: OptimizerStateType, info: _StateDictInfo) -> None:
    if not info.handle_optim:
        return
    for optim in optimizers:
        _init_optim_state(optim)
        if not state_dict:
            continue
        local = _split_optim_state_dict(model, optim, state_dict, info)
        if info.fsdp_modules:
            from tensorplay.distributed.fsdp import FullyShardedDataParallel

            with info.fsdp_context():
                local = FullyShardedDataParallel.optim_state_dict_to_load(
                    model,
                    optim,
                    local,
                    group=_get_fsdp_process_group(model, info),
                )
        _state_dict_fn(optim, "load_state_dict")(local)


def _unflatten_model_state_dict(model: Any, state_dict: dict[Any, Any]) -> dict[str, Any]:
    if not state_dict:
        return {}
    first = next(iter(state_dict))
    if isinstance(first, Module):
        result: dict[str, Any] = {}
        for submodule, values in state_dict.items():
            prefix = next((name for name, module in model.named_modules() if module is submodule), "")
            for key, value in values.items():
                result[f"{prefix}.{key}" if prefix else key] = value
        return result
    return cast(dict[str, Any], state_dict)


def get_model_state_dict(model: Any, *, submodules: set[Any] | None = None, options: StateDictOptions | None = None) -> dict[str, Any]:
    with _gc_context():
        info = _verify_options(model, (), False, submodules=submodules, options=options)
        result = _get_model_state_dict(model, info)
        _verify_state_dict(result, {}, info)
        return result


def get_optimizer_state_dict(model: Any, optimizers: Any, *, submodules: set[Any] | None = None, options: StateDictOptions | None = None) -> OptimizerStateType:
    optim_tuple = (optimizers,) if isinstance(optimizers, Optimizer) else tuple(optimizers)
    with _gc_context():
        info = _verify_options(model, optim_tuple, True, submodules=submodules, options=options)
        result = _get_optim_state_dict(model, optim_tuple, info)
        _verify_state_dict({}, result, info)
        return result


def get_state_dict(model: Any, optimizers: Any, *, submodules: set[Any] | None = None, options: StateDictOptions | None = None) -> tuple[dict[str, Any], OptimizerStateType]:
    optim_tuple = (optimizers,) if isinstance(optimizers, Optimizer) else tuple(optimizers)
    with _gc_context():
        info = _verify_options(model, optim_tuple, False, submodules=submodules, options=options)
        model_state = _get_model_state_dict(model, info)
        optim_state = _get_optim_state_dict(model, optim_tuple, info)
        _verify_state_dict(model_state, optim_state, info)
        return model_state, optim_state


def set_model_state_dict(model: Any, model_state_dict: dict[str, Any], *, options: StateDictOptions | None = None) -> Any:
    state = _unflatten_model_state_dict(model, model_state_dict)
    with _gc_context():
        info = _verify_options(model, (), False, options=options)
        _verify_state_dict(state, {}, info)
        return _load_model_state_dict(model, state, info)


def set_optimizer_state_dict(model: Any, optimizers: Any, optim_state_dict: OptimizerStateType, *, options: StateDictOptions | None = None) -> None:
    optim_tuple = (optimizers,) if isinstance(optimizers, Optimizer) else tuple(optimizers)
    with _gc_context():
        info = _verify_options(model, optim_tuple, True, options=options)
        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optim_tuple, optim_state_dict, info)


def set_state_dict(
    model: Any,
    optimizers: Any,
    *,
    model_state_dict: dict[str, Any],
    optim_state_dict: OptimizerStateType,
    options: StateDictOptions | None = None,
) -> Any:
    state = _unflatten_model_state_dict(model, model_state_dict)
    optim_tuple = (optimizers,) if isinstance(optimizers, Optimizer) else tuple(optimizers)
    with _gc_context():
        info = _verify_options(model, optim_tuple, not bool(state), options=options)
        _verify_state_dict(state, optim_state_dict, info)
        _load_optim_state_dict(model, optim_tuple, optim_state_dict, info)
        return _load_model_state_dict(model, state, info)


def _patch_model_state_dict(model: Any, *, options: StateDictOptions | None = None) -> None:
    def state_dict_call(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return get_model_state_dict(model, options=options)

    def load_state_dict_call(state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return set_model_state_dict(model, state_dict, options=options)

    model.state_dict = state_dict_call
    model.load_state_dict = load_state_dict_call
    _patched_state_dict.update({state_dict_call, load_state_dict_call})


def _patch_optimizer_state_dict(model: Any, *, optimizers: tuple[Any, ...], options: StateDictOptions | None = None) -> None:
    def state_dict_call(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return get_optimizer_state_dict(model, optimizers, options=options)

    def load_state_dict_call(state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return set_optimizer_state_dict(model, optimizers, state_dict, options=options)

    for optim in optimizers:
        optim.state_dict = state_dict_call
        optim.load_state_dict = load_state_dict_call
    _patched_state_dict.update({state_dict_call, load_state_dict_call})
