# mypy: allow-untyped-defs
r"""CUDA random-number generator helpers.

Seeding APIs and state serialization use the per-device generators of the
native runtime.
"""

from collections.abc import Iterable
from typing import Any

import random as _random

import tensorplay
from tensorplay import Tensor
from tensorplay._C import _cuda as _lcuda

from ._utils import _get_device_index
from . import _lazy_call, _lazy_init, current_device, device_count, is_initialized


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

def get_rng_state(device: int | str | Any = "cuda") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (tensorplay.Device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    index = _get_device_index(device, optional=True)
    return _lcuda.get_rng_state(index)


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: int | str | Any = "cuda") -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (ByteTensor): The desired state
        device (tensorplay.Device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., the current CUDA device).
    """
    _lazy_init()
    index = _get_device_index(device, optional=True)
    _lcuda.set_rng_state(new_state, index)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        _lcuda.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        from . import device as device_ctx

        prev_idx = current_device()
        try:
            for i in range(device_count()):
                with device_ctx(i):
                    _lcuda.manual_seed(seed)
        finally:
            from . import set_device

            set_device(prev_idx)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb():
        random_seed = _random.getrandbits(64) & 0x7FFFFFFF
        _lcuda.manual_seed(random_seed)

    _lazy_call(cb, seed=True)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """

    def cb():
        from . import device as device_ctx

        random_seed = _random.getrandbits(64) & 0x7FFFFFFF
        prev_idx = current_device()
        try:
            for i in range(device_count()):
                with device_ctx(i):
                    _lcuda.manual_seed(random_seed)
        finally:
            from . import set_device

            set_device(prev_idx)

    _lazy_call(cb)


def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    idx = current_device()
    return int(_lcuda.current_seed(idx))
