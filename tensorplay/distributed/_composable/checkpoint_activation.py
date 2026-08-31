from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, AbstractSet

import tensorplay.nn as nn
from tensorplay.distributed._composable.contract import _get_registry
from tensorplay.utils.checkpoint import checkpoint as _checkpoint_call

__all__ = ["checkpoint"]

_STATE_KEY = "__checkpoint_activation_state__"


@dataclass
class _CheckpointState:
    enabled: bool = True
    context_fn: Any = None
    kwargs: dict[str, Any] | None = None


def _state(module: nn.Module) -> _CheckpointState:
    registry = _get_registry(module)
    state = registry.get(_STATE_KEY)
    if state is None:
        state = _CheckpointState()
        registry[_STATE_KEY] = state
    return state


@contextmanager
def _no_hook(module: nn.Module, user_context: Any = None):
    state = _state(module)
    old = state.enabled
    state.enabled = False
    try:
        with user_context if user_context is not None else nullcontext():
            yield
    finally:
        state.enabled = old


def checkpoint(module: nn.Module, **kwargs: Any) -> nn.Module:
    if not isinstance(module, nn.Module):
        raise TypeError("checkpoint expects a module")
    allowed = {"use_reentrant", "preserve_rng_state", "context_fn", "determinism_check", "debug", "early_stop"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError("unexpected keyword arguments: " + ",".join(sorted(unknown)))
    if kwargs.get("use_reentrant", False):
        raise NotImplementedError("reentrant activation checkpointing is not supported")
    state = _state(module)
    if state.kwargs is not None:
        raise RuntimeError("activation checkpointing is already enabled for this module")
    state.kwargs = dict(kwargs)

    def pre_hook(current: nn.Module, args: tuple[Any, ...], call_kwargs: dict[str, Any]):
        current_state = _state(current)
        if not current_state.enabled:
            return None
        current_state._context = current_state.context_fn() if current_state.context_fn else None
        return args, call_kwargs

    def post_hook(current: nn.Module, args: tuple[Any, ...], output: Any):
        current_state = _state(current)
        current_state._context = None
        return output

    state.context_fn = kwargs.get("context_fn")
    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook, always_call=True)
    return module


checkpoint.state = staticmethod(_state)
