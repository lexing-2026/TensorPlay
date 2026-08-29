#
# Adaptations for tp: no typed-storage API (_alloc_storage/_free_storage are
# absent), no PackedSequence-aware recursion beyond tp.nn.utils.rnn, and
# stream-side copies fall back to plain .to().
import dataclasses
import traceback
from collections import OrderedDict
from collections.abc import Callable, Container
from typing import Any, Optional, overload, TypeVar

import tensorplay as tp

import tensorplay.distributed as dist


__all__ = []  # type: ignore[var-annotated]


def _pack_kwargs(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite).

    Inspiration: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: The first tuple element gives
        both positional args and kwarg values, where the positional args
        precede kwarg values and kwarg values are ordered consistently with the
        kwarg keys. The second tuple element gives the kwarg keys.
        The second tuple element's length is at most the first tuple element's length.
    """
    kwarg_keys: list[str] = []
    flat_args: list[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _cast_forward_inputs(
    dtype,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
    if dtype is None:
        return args, kwargs

    def cast_fn(x):
        if not x.is_floating_point() or x.dtype == dtype:
            return x

        return x.to(dtype)

    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))


def _unpack_kwargs(
    flat_args: tuple[Any, ...], kwarg_keys: tuple[str, ...]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """See _pack_kwargs."""
    if len(kwarg_keys) > len(flat_args):
        raise AssertionError(f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}")
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
    return args, kwargs


S = TypeVar("S", dict, list, tuple)
T = TypeVar("T")


@overload
def _recursive_to(
    inputs: S, target_device, use_side_stream_for_tensor_copies: bool
) -> list[S]: ...


def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    r"""Recursively moves input to the target_device."""

    def to_map(obj):
        if isinstance(obj, tp.Tensor):
            device = obj.device
            target = tp.device(target_device)
            if device == target:
                return (obj,)
            # tp has no side-stream copy machinery yet; a plain .to() keeps
            # stream ordering safe on the current stream.
            return (obj.to(target_device),)

        from tensorplay.nn.parallel.scatter_gather import _is_namedtuple

        def _handle_container(obj, elements, make_container):
            mapped = list(map(to_map, elements))
            # Preserve object identity when all elements are unchanged (single-device case)
            if all(len(m) == 1 for m in mapped):
                transformed = [m[0] for m in mapped]
                if all(t is o for t, o in zip(transformed, elements)):
                    return [obj]
                return [make_container(transformed)]
            return [make_container(args) for args in zip(*mapped)]

        if _is_namedtuple(obj):
            return _handle_container(obj, obj, lambda x: type(obj)(*x))
        if isinstance(obj, tuple) and len(obj) > 0:
            return _handle_container(obj, obj, tuple)
        if isinstance(obj, list) and len(obj) > 0:
            return _handle_container(obj, obj, list)
        if isinstance(obj, dict) and len(obj) > 0:
            keys = list(obj.keys())
            return _handle_container(
                obj, obj.values(), lambda v: type(obj)(zip(keys, v))
            )
        return [obj]

    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


Q = TypeVar("Q")
R = TypeVar("R", dict, list, tuple, set, OrderedDict, Any)


@overload
def _apply_to_tensors(
    fn: Callable[[Any], Q], container: tp.Tensor
) -> Q: ...


@overload
def _apply_to_tensors(fn: Callable[[Any], Any], container: R) -> R: ...


def _apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x):
        from tensorplay.nn.parallel.scatter_gather import _is_namedtuple

        if isinstance(x, tp.Tensor):
            return fn(x)
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            changes = {
                f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
            }
            return dataclasses.replace(dc, **changes)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)


def _to_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_device,
    use_side_stream_for_tensor_copies: bool,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    moved_inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return tuple(moved_inputs), tuple(moved_kwargs)


def _verify_param_shape_across_processes(
    process_group,
    tensors,
    logger: Optional["dist.DistLogger"] = None,  # noqa: F821
):
    try:
        return dist._verify_params_across_processes(process_group, tensors, logger)
    except AttributeError:
        # tp fallback: MIN/MAX allreduce over sizes.
        from tensorplay.nn.parallel.distributed import (
            _verify_param_shape_across_processes as _impl,
        )

        return _impl(process_group, tensors)


def _sync_module_states(
    module,
    process_group,
    broadcast_bucket_size: int,
    src: int,
    params_and_buffers_to_ignore: Container[str],
    broadcast_buffers: bool = True,
) -> None:
    """
    Sync ``module``'s parameters and buffers state.

    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
    module_states: list[tp.Tensor] = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())

    _sync_params_and_buffers(process_group, module_states, broadcast_bucket_size, src)


def _sync_params_and_buffers(
    process_group,
    module_states,
    broadcast_bucket_size: int,
    src: int,
) -> None:
    """Synchronize ``module_states`` across all processes by broadcasting them from rank ``src``."""
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )


def _replace_by_prefix(
    state_dict: dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": tp.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": tp.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _get_root_modules(modules: list) -> list:
    """
    Returns the modules in ``modules`` that are root modules (i.e.
    parent-less) with respect to the set ``modules``. In other words, these
    are the modules in ``modules`` that are the not child of any other
    module in ``modules``.
    """
    root_modules: list = []
    module_to_modules: dict = {
        module: set(module.modules()) for module in modules
    }
    for candidate_module in modules:
        is_root_module = True
        for module, _modules in module_to_modules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in _modules
            )
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.append(candidate_module)
    return root_modules
