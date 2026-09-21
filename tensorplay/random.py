"""

CPU RNG: for a given seed, ``tensorplay`` produces the same random sequences
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
    "thread_safe_generator",
]


def thread_safe_generator():
    """Returns a thread-safe random number generator for use in DataLoader workers.

    This function provides a convenient way for transforms and user code to use
    thread-safe random number generation without manually checking worker context.

    When called in a DataLoader thread worker, returns the worker's thread-local
    :class:`tensorplay.Generator`. When called in the main process or process workers,
    returns ``None`` (which causes operations to use the default global RNG).

    Example:
        >>> from tensorplay.random import thread_safe_generator
        >>> generator = thread_safe_generator()
        >>> tensorplay.randint(0, 10, (5,), generator=generator)
    """
    from .utils.data import get_worker_info

    worker_info = get_worker_info()
    if (
        worker_info is not None
        and worker_info.worker_method == "thread"
        and worker_info.rng is not None
    ):
        return worker_info.rng.torch_generator
    return None


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller="fork_rng", _devices_kw="devices"):
    """Forks the RNG state: code inside the context gets a pristine RNG.

    Saves the CPU RNG state and the RNG state of every CUDA device listed in
    ``devices`` on entry and restores them on exit, so random operations
    inside the block do not advance the outer streams.  ``devices=[]`` forks
    only the CPU generator.
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
    device_states = []
    if devices:
        import tensorplay.cuda as _cuda

        for device in devices:
            device_states.append((device, _cuda.get_rng_state(device)))
    cpu_state = get_rng_state()
    try:
        yield
    finally:
        set_rng_state(cpu_state)
        if device_states:
            import tensorplay.cuda as _cuda

            for device, state in device_states:
                _cuda.set_rng_state(state, device)
