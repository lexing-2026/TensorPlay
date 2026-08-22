"""Random number generation utilities, mirroring ``torch.random``.

The underlying generator is a bit-compatible port of PyTorch's MT19937-based
CPU RNG: for a given seed, ``tensorplay`` produces the same random sequences
as ``torch`` for the ops sharing torch's transformation formulas.
"""

import contextlib

from ._C import (
    default_generator,
    get_rng_state,
    initial_seed,
    manual_seed,
    seed,
    set_rng_state,
)

__all__ = [
    "default_generator",
    "fork_rng",
    "get_rng_state",
    "initial_seed",
    "manual_seed",
    "seed",
    "set_rng_state",
]


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller="fork_rng", _devices_kw="devices"):
    """Forks the RNG state: code inside the context gets a pristine RNG.

    Saves the CPU RNG state on entry and restores it on exit, so random
    operations inside the block do not advance the outer stream (mirrors
    ``torch.random.fork_rng``). CUDA devices are not forked; pass
    ``devices=[]`` explicitly to acknowledge that (same contract as torch).
    """
    if not enabled:
        yield
        return
    if devices is None:
        raise RuntimeError(
            f"{_caller} was called without an explicit value for the {_devices_kw} "
            "argument, which is no longer allowed since it defaults to forking all "
            "CUDA devices. Pass devices=[] to only fork the CPU RNG."
        )
    cpu_state = get_rng_state()
    try:
        yield
    finally:
        set_rng_state(cpu_state)
