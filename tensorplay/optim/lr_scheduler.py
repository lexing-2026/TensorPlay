import abc
import math
import warnings
from typing import List, Optional, Any

from .optimizer import Optimizer


class _LRScheduler(abc.ABC):
    r"""
    Base class for learning rate schedulers.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Attributes:
        optimizer (Optimizer): The wrapped optimizer.
        base_lrs (List[float]): List of initial learning rates for each parameter group.
        last_epoch (int): The index of last epoch.
        verbose (bool): Whether to print update messages.
        _last_lr (Optional[List[float]]): List of last learning rates for each parameter group.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')

        self.optimizer = optimizer
        self.verbose = verbose
        self._last_lr = None

        if last_epoch == -1:
            for group in optimizer.param_groups:
                if 'initial_lr' in group:
                    warnings.warn(
                        "param_group already has 'initial_lr' key, overwriting it",
                        UserWarning
                    )
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        f"param 'initial_lr' is not specified in param_groups[{i}] "
                        "when resuming an optimizer"
                    )

        self.base_lrs = [g['initial_lr'] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        # Step once to initialize
        self.step()

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the state of the scheduler as a dict.

        Includes all attributes except 'optimizer' to avoid circular references.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads the scheduler state.

        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a call to :meth:`state_dict`.

        Raises:
            ValueError: If the number of base_lrs in state_dict does not match the current optimizer's param_groups.
        """
        if 'base_lrs' in state_dict and len(state_dict['base_lrs']) != len(self.optimizer.param_groups):
            raise ValueError(
                f"Expected {len(self.optimizer.param_groups)} base_lrs in state_dict, "
                f"got {len(state_dict['base_lrs'])}"
            )
        if 'last_epoch' in state_dict and not isinstance(state_dict['last_epoch'], int):
            raise TypeError(f"last_epoch must be an integer, got {type(state_dict['last_epoch'])}")
        self.__dict__.update(state_dict)

    @abc.abstractmethod
    def get_lr(self) -> List[float]:
        """
        Compute the learning rate for the current epoch.

        Returns:
            List[float]: Learning rates for each parameter group. Must have the same length as param_groups.
        """
        raise NotImplementedError

    def get_last_lr(self) -> List[float]:
        """
        Return last computed learning rates by the scheduler.

        Raises:
            RuntimeError: If the scheduler has not stepped yet.
        """
        if self._last_lr is None:
            raise RuntimeError("get_last_lr() called before first step()")
        return self._last_lr

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step the scheduler to update learning rates.

        Args:
            epoch (Optional[int]): The epoch index to set. If None, increment last_epoch by 1.

        Raises:
            ValueError: If epoch is a non-integer or negative value, or less than current last_epoch.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            if not isinstance(epoch, int):
                raise TypeError(f"epoch must be an integer, got {type(epoch)}")
            if epoch < 0:
                raise ValueError(f"epoch must be non-negative, got {epoch}")
            if epoch < self.last_epoch:
                warnings.warn(
                    f"Setting epoch {epoch} which is less than current last_epoch {self.last_epoch}. "
                    "This may cause unexpected learning rate changes.",
                    UserWarning
                )
            self.last_epoch = epoch

        values = self.get_lr()
        if len(values) != len(self.optimizer.param_groups):
            raise ValueError(
                f"get_lr() returned {len(values)} values, "
                f"but optimizer has {len(self.optimizer.param_groups)} param groups"
            )
        for param_group, lr in zip(self.optimizer.param_groups, values):
            if not isinstance(lr, (int, float)) or lr < 0:
                raise ValueError(f"Invalid learning rate {lr} for param group")
            param_group['lr'] = lr
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class StepLR(_LRScheduler):
    r"""
    Set the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class MultiStepLR(_LRScheduler):
    r"""
    Set the learning rate of each parameter group to the initial lr
    decayed by gamma once the number of epoch reaches one of the milestones.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (set of int): Set of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = set(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class ExponentialLR(_LRScheduler):
    r"""
    Set the learning rate of each parameter group to the initial lr
    decayed by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(_LRScheduler):
    r"""
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{max}` is the number of epochs to decay. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        t_max (int): Maximum number of epochs.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self, optimizer, t_max, eta_min=0, last_epoch=-1, verbose=False):
        self.t_max = t_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.t_max) % (2 * self.t_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.t_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.t_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.t_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class ReduceLROnPlateau:
    r"""
    Base class for reducing learning rate when a metric has stopped
    improving. Models often benefit from reducing the learning rate by a
    factor of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number of epochs,
    the learning rate is reduced.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = None
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {epoch:5d}: reducing learning rate of group {i} to {new_lr:.4e}.')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
