# Copyright 2022 Cruise LLC
import logging
import warnings
from collections import OrderedDict
from collections.abc import Iterable

import tensorplay.distributed as dist
import tensorplay.distributed.algorithms.model_averaging.averagers as averagers
import tensorplay.distributed.algorithms.model_averaging.utils as utils


logger = logging.getLogger(__name__)


class HierarchicalModelAverager(averagers.ModelAverager):
    r"""
    Runs hierarchical model averaging (`hierarchical SGD <https://arxiv.org/pdf/2010.12998.pdf>`_).

    Process groups of different sizes are organized in a hierarchy, and they average parameters
    by using different periods concurrently after the warm-up stage.
    This is an extension of :class:`~tensorplay.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`
    that supports `post-local SGD <https://arxiv.org/abs/1808.07217>`_, which essentially only supports
    a two-level hierarchy: the intra-machine level and the global level, where the intra-machine
    level is usually embedded in :meth:`~tensorplay.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook`.
    Similarly, the process groups within this class do not have such an intra-machine process
    subgroup, which should be embedded by the post-local SGD communication hook instead.

    Args:
        period_group_size_dict: An ordered dict mapping keys of model averaging period to
                                process group size, used for initializing process groups of
                                different sizes in a hierarchy to average parameters concurrently.
        warmup_steps (int): The number of warm-up steps. During this stage, model averaging is skipped.
        process_group (ProcessGroup, optional): The overall process group containing all the processes that runs model averaging.

    .. warning ::
        The last group size in the dict must be the size of the provided ``process_group``,
        which indicates model averaging at the highest level of the hierarchy.
        If ``process_group`` is not provided, then the last group size should be equal to the world size.

    .. warning ::
        `HierarchicalModelAverager` is experimental and subject to change.
    """

    def __init__(self, period_group_size_dict=None, warmup_steps=0, process_group=None):
        super().__init__(process_group)
        if not period_group_size_dict:
            raise ValueError("Arg ``period_group_size_dict`` must not be empty.")
        self._periods = list(period_group_size_dict.keys())
        if self._periods[0] <= 0:
            raise ValueError(
                "The minimum period in arg ``period_group_size_dict`` must be a positive value."
            )
        elif self._periods[-1] == 1:
            warnings.warn(
                "When the maximum period in arg ``period_group_size_dict`` is 1, "
                "no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case.",
                stacklevel=2,
            )
        overall_group_size = dist.get_world_size(group=self.process_group)
        if list(period_group_size_dict.values())[-1] != overall_group_size:
            raise ValueError(
                f"The last value in arg ``period_group_size_dict`` {list(period_group_size_dict.values())[-1]} "
                f"must be equal to the size of arg ``process_group`` {overall_group_size}."
            )

        self.period_process_group_dict = OrderedDict()
        logger.info("Model averaging hierarchy:")
        for period, group_size in period_group_size_dict.items():
            logger.info(
                "\tEach group that has %s processes average parameters every %s iterations, "
                "if no higher-level averaging.",
                group_size,
                period,
            )
            if group_size != overall_group_size:
                self.period_process_group_dict[period], _ = dist.new_subgroups(
                    group_size=group_size, group=self.process_group
                )
            else:
                self.period_process_group_dict[period] = self.process_group

        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps

    def _find_process_group(self):
        """
        Return a process group as the value of a ``period_process_group_dict`` entry.

        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        then the returned process group is the one corresponding to the largest period,
        since this process group will be used for averaging parameters at this ``step``.
        Returns ``None`` if not found.
        """
        for period in reversed(self._periods):
            if self.step % period == 0:
                return self.period_process_group_dict[period]
        return None

    def average_parameters(
        self,
        params: Iterable,
    ):
        """
        Averages parameters or parameter groups of an optimizer.

        Averaging only occurs if ``step`` is no less than ``warmup_steps``
        and it can be divided by a period in the keys of ``period_process_group_dict``,
        where ``step`` is increased by 1 at each iteration in the training loop.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        only the largest period is used, and the corresponding process group is used for averaging parameters.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.
        """
        if self.step >= self.warmup_steps:
            group = self._find_process_group()
            if group is not None:
                utils.average_parameters_or_parameter_groups(params, group)
        self.step += 1
