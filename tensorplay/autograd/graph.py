"""autograd.graph -- context managers around graph bookkeeping.

pack hooks run when tensors are stashed via ``save_for_backward``; the
matching unpack hook runs when ``saved_tensors`` is later read (typically
inside ``backward``, after the pack context has exited).
"""

from contextlib import contextmanager

__all__ = ["saved_tensors_hooks"]

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
    try:
        yield
    finally:
        _hook_stack.pop()


def _current_pair():
    return _hook_stack[-1] if _hook_stack else None
