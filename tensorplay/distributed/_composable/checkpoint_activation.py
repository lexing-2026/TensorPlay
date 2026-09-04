from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

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
    original_forward: Callable[..., Any] | None = None
    wrapped_forward: Callable[..., Any] | None = None
    _ac_generator: Any = None


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

    state.context_fn = kwargs.get("context_fn")
    state.kwargs = dict(kwargs)
    state.kwargs.setdefault("use_reentrant", False)

    original_forward = module.forward
    state.original_forward = original_forward

    def checkpointed_forward(*args: Any, **call_kwargs: Any) -> Any:
        current_state = _state(module)
        if not current_state.enabled:
            return original_forward(*args, **call_kwargs)

        def invoke(
            packed_args: tuple[Any, ...],
            packed_kwargs: dict[str, Any],
        ) -> Any:
            with _no_hook(module):
                return original_forward(*packed_args, **packed_kwargs)

        invoke.parameters = module.parameters
        return _checkpoint_call(
            invoke,
            tuple(args),
            dict(call_kwargs),
            **dict(current_state.kwargs or {}),
        )

    state.wrapped_forward = checkpointed_forward
    module.__dict__["forward"] = checkpointed_forward
    return module


checkpoint.state = staticmethod(_state)
