"""autograd.graph -- context managers around graph bookkeeping.

pack hooks run when tensors are stashed via ``save_for_backward``; the
matching unpack hook runs when ``saved_tensors`` is later read (typically
inside ``backward``, after the pack context has exited).
"""

from contextlib import contextmanager

import tensorplay
import tensorplay._C._autograd as _native_autograd

__all__ = ["saved_tensors_hooks", "save_on_cpu"]

# Active (pack_fn, unpack_fn) pairs, innermost last.  The pair captured at
# save time travels with the node's context so unpack works post-exit.
_hook_stack: list = []


@contextmanager
def saved_tensor_hooks(pack_hook, unpack_hook):
    """Context manager that installs a (pack, unpack) pair for every tensor
    saved via :meth:`~tensorplay.autograd.Function`'s ``save_for_backward``
    while active.

    * ``pack_hook(tensor)`` runs once per saved tensor at save time and may
      return any object (e.g. a CPU copy, a key into external storage).
    * ``unpack_hook(packed)`` runs when ``ctx.saved_tensors`` reads it back
      and must return an equivalent :class:`tensorplay.Tensor`.

    If either hook raises, the exception is surfaced from the corresponding
    operation.
    """
    _hook_stack.append((pack_hook, unpack_hook))
    native_push = getattr(_native_autograd, "_push_saved_tensors_hooks", None)
    native_pop = getattr(_native_autograd, "_pop_saved_tensors_hooks", None)
    native_active = False
    try:
        if native_push is not None:
            native_push(pack_hook, unpack_hook)
            native_active = True
        yield
    finally:
        if native_active:
            native_pop()
        _hook_stack.pop()


saved_tensors_hooks = saved_tensor_hooks


@contextmanager
def save_on_cpu(pin_memory: bool = False, device_type: str = "cuda"):
    """Store saved values in host memory and restore their original device."""
    del device_type

    def pack(tensor):
        device = tensor.device
        if str(device.type) == "cpu":
            return tensor, device
        host = tensor.to(tensorplay.Device(tensorplay.DeviceType.CPU),
                         non_blocking=False, copy=True)
        if pin_memory:
            host = host.pin_memory()
        return host, device

    def unpack(packed):
        host, device = packed
        if str(device.type) == "cpu":
            return host
        return host.to(device, non_blocking=pin_memory, copy=True)

    with saved_tensor_hooks(pack, unpack):
        yield


def _current_pair():
    return _hook_stack[-1] if _hook_stack else None
