# mypy: allow-untyped-defs
"""Device-agnostic random-number helpers for the current accelerator.

Seeds and generator states are read and written through the active
accelerator backend (the GPU backend in this build). ``device``
accepts a device object, a spelling such as ``"cuda:1"``, an integer
index, or ``None`` for the current device.
"""

from collections.abc import Iterable
from typing import Any

from ._utils import _require_accelerator, _resolve_device

__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]


def _backend():
    _require_accelerator()
    import tensorplay.cuda.random as _rng

    return _rng


def _with_device(device: Any, fn):
    index = _resolve_device(device, optional=True)
    if index < 0:
        return fn()
    import tensorplay.cuda as _cuda

    prev = int(_cuda.current_device())
    try:
        _cuda.set_device(index)
        return fn()
    finally:
        _cuda.set_device(prev)


def get_rng_state(device: Any = None):
    """Generator state of the given accelerator device."""
    return _with_device(device, lambda: _backend().get_rng_state())


def get_rng_state_all() -> list:
    """Generator states of every accelerator device."""
    return _backend().get_rng_state_all()


def set_rng_state(new_state, device: Any = None) -> None:
    """Set the generator state of the given accelerator device."""
    _require_accelerator()
    import tensorplay.cuda.random as _rng

    index = _resolve_device(device, optional=True)
    if index < 0:
        _rng.set_rng_state(new_state)
    else:
        _rng.set_rng_state(new_state, index)


def set_rng_state_all(new_states: Iterable) -> None:
    """Set the generator states of every accelerator device."""
    _backend().set_rng_state_all(new_states)


def manual_seed(seed: int) -> None:
    """Seed the generator of the current accelerator device."""
    _backend().manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    """Seed the generators of every accelerator device."""
    _backend().manual_seed_all(seed)


def seed() -> None:
    """Reseed the generator of the current accelerator device."""
    _backend().seed()


def seed_all() -> None:
    """Reseed the generators of every accelerator device."""
    _backend().seed_all()


def initial_seed(device: Any = None) -> int:
    """Initial seed of the generator for the given accelerator device."""
    return _with_device(device, lambda: _backend().initial_seed())
