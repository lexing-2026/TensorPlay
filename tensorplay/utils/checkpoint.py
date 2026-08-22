"""``torch.utils.checkpoint`` compatibility.

torchvision's densenet accepts ``memory_efficient=True`` and wraps block
computation in ``torch.utils.checkpoint.sequential``.  TensorPlay does not
yet implement activation recomputation, so checkpoint executes the function
eagerly — numerically identical, without the memory savings.  The call
signature mirrors torch so model code needs no changes.
"""

import warnings
from typing import Any, Callable, Iterable, Iterator, Sequence

import tensorplay as tp


def checkpoint(function: Callable[..., Any], *args: Any, use_reentrant: bool = True,
               context_fn: Callable[[], Any] = None,
               determinism_check: str = "default",
               debug: bool = False, **kwargs: Any) -> Any:
    """Runs ``function(*args, **kwargs)`` eagerly (no recomputation).

    Mirrors torch.utils.checkpoint.checkpoint's signature; emits a warning
    once when grad is enabled because backward will re-run the graph normally.
    """
    if any(isinstance(a, tp.Tensor) and a.requires_grad for a in args):
        warnings.warn(
            "tensorplay.utils.checkpoint: activation recomputation is not "
            "implemented; running eagerly. Memory usage will not be reduced."
        )
    return function(*args, **kwargs)


def checkpoint_sequential(
    function: Callable[..., Any],
    chunks: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """torch.utils.checkpoint.checkpoint_sequential compatibility.

    With ``chunks`` segments the sequential would be evaluated in chunks with
    recomputation; TensorPlay runs it as one eager segment.
    """
    if isinstance(args[-1], tp.Tensor):
        warnings.warn(
            "tensorplay.utils.checkpoint_sequential: activation recomputation "
            "is not implemented; running eagerly."
        )
    return function(*args, **kwargs)


def set_checkpoint_early_stop(enabled: bool) -> None:
    """No-op for API parity."""


class CheckpointPolicy:
    pass


class checkpoint_wrapper:
    def __init__(self, m, *args: Any, **kwargs: Any) -> None:
        self._module = m

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)
