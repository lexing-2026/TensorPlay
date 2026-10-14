"""Cubic sparsity scheduler."""

import warnings

from .base_scheduler import BaseScheduler

__all__ = ["CubicSL"]


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CubicSL(BaseScheduler):
    """Schedule sparsity along a cubic curve between initial and final levels."""

    def __init__(self, sparsifier, init_sl=0.0, init_t=0, delta_t=10, total_t=100,
                 initially_zero=False, last_epoch=-1, verbose=False):
        self.sparsifier = sparsifier
        self.init_sl = self._make_sure_a_list(init_sl)
        self.init_t = self._make_sure_a_list(init_t)
        self.delta_t = self._make_sure_a_list(delta_t)
        self.total_t = self._make_sure_a_list(total_t)
        self.initially_zero = self._make_sure_a_list(initially_zero)
        super().__init__(sparsifier, last_epoch, verbose)

    @staticmethod
    def sparsity_compute_fn(s_0, s_f, t, t_0, dt, n, initially_zero=False):
        if initially_zero and t < t_0:
            return 0
        s_t = s_f + (s_0 - s_f) * (1.0 - (t - t_0) / (dt * n)) ** 3
        s_t = _clamp(s_t, s_0, s_f)
        return s_t

    def get_sl(self):
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        return [
            self.sparsity_compute_fn(
                s_0=initial_sparsity,
                s_f=final_sparsity,
                t=self.last_epoch,
                t_0=initial_epoch,
                dt=delta_epoch,
                n=interval_epochs,
                initially_zero=initially_zero,
            )
            for initial_sparsity, final_sparsity, initial_epoch, delta_epoch, interval_epochs, initially_zero in zip(
                self.init_sl, self.base_sl, self.init_t, self.delta_t, self.total_t,
                self.initially_zero,
            )
        ]