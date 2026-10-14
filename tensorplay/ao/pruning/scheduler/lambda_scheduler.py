"""Lambda sparsity scheduler."""

import warnings
from collections.abc import Callable

from .base_scheduler import BaseScheduler

__all__ = ["LambdaSL"]


class LambdaSL(BaseScheduler):
    """Scale the base sparsity level of each group by ``sl_lambda(epoch)``."""

    def __init__(self, sparsifier, sl_lambda, last_epoch=-1, verbose=False):
        self.sparsifier = sparsifier
        if not isinstance(sl_lambda, list) and not isinstance(sl_lambda, tuple):
            self.sl_lambdas = [sl_lambda] * len(sparsifier.groups)
        else:
            if len(sl_lambda) != len(sparsifier.groups):
                raise ValueError(f"Expected {len(sparsifier.groups)} lr_lambdas, but got {len(sl_lambda)}")
            self.sl_lambdas = list(sl_lambda)
        super().__init__(sparsifier, last_epoch, verbose)

    def get_sl(self) -> list[float]:
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        return [
            base_sl * lmbda(self.last_epoch)
            for lmbda, base_sl in zip(self.sl_lambdas, self.base_sl)
        ]