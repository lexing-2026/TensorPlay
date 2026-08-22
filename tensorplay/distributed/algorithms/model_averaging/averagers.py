# Ported from torch/distributed/algorithms/model_averaging/averagers.py.
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.distributed.algorithms.model_averaging.utils as utils


__all__ = ["ModelAverager", "PeriodicModelAverager"]


def _not_none(x):
    if x is None:
        raise ValueError("Expected non-None value")
    return x


class ModelAverager(ABC):
    r"""Base class for all model averagers.

    Args:
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`tensorplay.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """

    def __init__(self, process_group: dist.ProcessGroup | None = None):
        self.process_group = (
            process_group if process_group is not None else _not_none(dist.GroupMember.WORLD)
        )
        self.step = 0

    @abstractmethod
    def average_parameters(self, params):
        raise NotImplementedError


class PeriodicModelAverager(ModelAverager):
    r"""
    Averages parameters periodically after the warm-up stage.

    This can be used for running `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    by running :class:`~tensorplay.nn.DistributedDataParallel` (DDP)
    using the subgroups created by :meth:`~tensorplay.distributed.new_subgroups`.

    Args:
        period (int): The number of steps per model averaging.
                      Usually the period should be greater than ``1`` to reduce the communication cost.
                      Otherwise, only DDP needs to be used.
        warmup_steps (int): The number of warm-up steps. During this stage,
                            model averaging is skipped.
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`tensorplay.distributed.init_process_group`,
                       will be used. (default: ``None``)

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> averager = averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>> for step in range(0, 200):
        >>>    optimizer.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    optimizer.step()
        >>>    # Will average model parameters globally every 4 steps.
        >>>    averager.average_parameters(model.parameters())
    """

    def __init__(
        self, period, warmup_steps=0, process_group: dist.ProcessGroup | None = None
    ):
        super().__init__(process_group)
        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps
        if period < 1:
            raise ValueError("Arg ``period`` must be a positive value.")
        elif period == 1:
            warnings.warn(
                "When period is 1, no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case.",
                stacklevel=2,
            )
        self.period = period

    def average_parameters(
        self,
        params: Iterable[tp.nn.Parameter] | Iterable[dict[str, tp.nn.Parameter]],
    ):
        """
        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``.

        Can be divided by ``period``, where ``step`` is increased by 1
        at each iteration in the training loop.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.

        """
        if (
            self.step >= self.warmup_steps
            and (self.step - self.warmup_steps) % self.period == 0
        ):
            utils.average_parameters_or_parameter_groups(
                params, _not_none(self.process_group)
            )
        self.step += 1
