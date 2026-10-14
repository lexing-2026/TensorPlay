"""Spike monitoring utilities.

:class:`SpikeMonitor` attaches to any :class:`tensorplay.nn.Module`, records
every spike tensor the module produces, and exposes firing-rate statistics.
It is the training-time counterpart of the fused neuron operators: the
monitor itself is pure Python and adds no kernels.
"""

from typing import List, Optional

from tensorplay import Tensor

__all__ = ["SpikeMonitor"]


class SpikeMonitor:
    """Record and summarize the spike output of a module.

    Args:
        module: the module to watch.  May be ``None`` and attached later
            with :meth:`attach`.

    The monitor registers a forward hook on ``module`` and stores a detached
    copy of every spike tensor the module returns.  If the module's forward
    returns ``(spikes, v_seq)`` (``return_v_seq=True``), only the first
    element is recorded.
    """

    def __init__(self, module=None) -> None:
        self.module = module
        self._records: List[Tensor] = []
        self._handle = None
        if module is not None:
            self.attach(module)

    def attach(self, module) -> None:
        """Start monitoring ``module``, replacing any previous target."""
        self.detach()
        self.module = module
        self._handle = module.register_forward_hook(self._hook)

    def detach(self) -> None:
        """Remove the forward hook and stop recording."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.module = None

    def reset(self) -> None:
        """Clear all recorded spike tensors."""
        self._records.clear()

    @staticmethod
    def _as_spikes(output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _hook(self, module, args, output):
        spikes = self._as_spikes(output)
        if spikes is not None:
            self._records.append(spikes.detach())

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"SpikeMonitor(module={type(self.module).__name__ if self.module else None}, "
            f"records={len(self._records)})"
        )

    def last_spike(self) -> Optional[Tensor]:
        """Return the most recently recorded spike tensor, if any."""
        if not self._records:
            return None
        return self._records[-1]

    def spike_rate(self) -> float:
        """Average firing rate over all recorded spike tensors."""
        if not self._records:
            return 0.0
        return sum(float(r.mean()) for r in self._records) / len(self._records)

    def firing_rate(self, dim: int = 0):
        """Per-neuron mean firing rate.

        Averages the recorded spike tensors along ``dim`` (the time axis for
        ``[T, ...]`` spike trains) and then over all records.  Returns ``None``
        when nothing has been recorded.
        """
        if not self._records:
            return None
        rates = [r.mean(dim=dim) for r in self._records]
        acc = rates[0]
        for r in rates[1:]:
            acc = acc + r
        return acc / len(rates)

    def total_spikes(self) -> int:
        """Total number of spikes across all records."""
        if not self._records:
            return 0
        return int(sum(float(r.sum()) for r in self._records))