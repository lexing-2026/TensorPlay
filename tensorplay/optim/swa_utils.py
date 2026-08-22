# mypy: allow-untyped-defs
r"""Implementation for Stochastic Weight Averaging implementation."""

from __future__ import annotations

import itertools
import math
import warnings
from copy import deepcopy
from typing import Any, cast, Literal, TYPE_CHECKING, TypeAlias
from typing_extensions import override

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.autograd.grad_mode import no_grad
from tensorplay.nn import Module
from tensorplay.nn.modules.batchnorm import _BatchNorm

from .lr_scheduler import _LRScheduler
from tensorplay.utils._foreach_utils import (
    _get_foreach_kernels_supported_devices,
    _group_tensors_by_device_and_dtype,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .optimizer import Optimizer


__all__ = [
    "AveragedModel",
    "update_bn",
    "SWALR",
    "get_ema_multi_avg_fn",
    "get_swa_multi_avg_fn",
    "get_ema_avg_fn",
    "get_swa_avg_fn",
]


PARAM_LIST: TypeAlias = tuple[Tensor, ...] | list[Tensor]


def get_ema_multi_avg_fn(decay=0.999):
    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @no_grad()
    def ema_update(
        ema_param_list: PARAM_LIST, current_param_list: PARAM_LIST, _
    ) -> None:
        # foreach lerp only handles float and complex
        if ema_param_list[0].is_floating_point() or tp.is_complex(ema_param_list[0]):
            tp._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    return ema_update


def get_swa_multi_avg_fn():
    @no_grad()
    def swa_update(
        averaged_param_list: PARAM_LIST,
        current_param_list: PARAM_LIST,
        num_averaged: Tensor | int,
    ) -> None:
        # foreach lerp only handles float and complex
        if (
            averaged_param_list[0].is_floating_point()
            or tp.is_complex(averaged_param_list[0])
        ):
            tp._foreach_lerp_(
                averaged_param_list,
                current_param_list,
                cast(float, 1 / (num_averaged + 1)),
            )
        else:
            diffs = tp._foreach_sub(current_param_list, averaged_param_list)
            if isinstance(num_averaged, Tensor):
                tp._foreach_addcdiv_(
                    averaged_param_list,
                    diffs,
                    [num_averaged + 1] * len(averaged_param_list),
                )
            else:
                tp._foreach_add_(
                    averaged_param_list, diffs, alpha=1.0 / (num_averaged + 1)
                )

    return swa_update


def get_ema_avg_fn(decay=0.999):
    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @no_grad()
    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update


def get_swa_avg_fn():
    @no_grad()
    def swa_update(
        averaged_param: Tensor, current_param: Tensor, num_averaged: Tensor | int
    ):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)

    return swa_update


class AveragedModel(Module):
    """A model that maintains a running SWA or EMA copy of another model."""

    n_averaged: Tensor

    def __init__(
        self,
        model: Module,
        device: int | tp.device | None = None,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, Tensor | int], None]
        | None = None,
        use_buffers=False,
    ) -> None:
        super().__init__()
        if avg_fn is not None and multi_avg_fn is not None:
            raise AssertionError("Only one of avg_fn and multi_avg_fn should be provided")
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer(
            "n_averaged", tp.tensor(0, dtype=tp.int64, device=device)
        )
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: Module) -> None:
        """Update model parameters."""
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )
        self_param_detached: list[Tensor | None] = []
        model_param_detached: list[Tensor | None] = []
        copy_param = bool(self.n_averaged == 0)
        for p_averaged, p_model in zip(self_param, model_param, strict=False):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if copy_param:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped_tensors = _group_tensors_by_device_and_dtype(
                    [self_param_detached, model_param_detached]
                )
                for (device, _), (
                    [self_params, model_params],
                    _,
                ) in grouped_tensors.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(
                            self_params,
                            model_params,
                            self.n_averaged.to(device),
                        )
                    elif (
                        device is not None
                        and device.type in _get_foreach_kernels_supported_devices()
                    ):
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(
                            self_params, model_params, self.n_averaged.to(device)
                        )
                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        for p_averaged, p_model in zip(
                            self_params, model_params, strict=True
                        ):
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
            else:
                for p_averaged, p_model in zip(
                    self_param_detached, model_param_detached, strict=True
                ):
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    p_averaged.detach().copy_(
                        self.avg_fn(p_averaged.detach(), p_model, n_averaged)
                    )

        if not self.use_buffers:
            for b_swa, b_model in zip(
                self.module.buffers(), model.buffers(), strict=True
            ):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1


@no_grad()
def update_bn(
    loader: Iterable[Any],
    model: Module,
    device: int | tp.device | None = None,
) -> None:
    momenta = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum
    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta:
        module.momentum = None
    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)
        model(input)
    for bn_module in momenta:
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(_LRScheduler):
    """Anneal each optimizer learning rate to a fixed SWA learning rate."""

    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs=10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch=-1,
    ) -> None:
        if isinstance(swa_lr, (list, tuple)):
            if len(swa_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"swa_lr has {len(swa_lr)} values, but optimizer has "
                    f"{len(optimizer.param_groups)} parameter groups"
                )
            swa_lrs = list(swa_lr)
        else:
            swa_lrs = [swa_lr] * len(optimizer.param_groups)
        for group, group_swa_lr in zip(
            optimizer.param_groups, swa_lrs, strict=True
        ):
            group["swa_lr"] = group_swa_lr
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(
                "anneal_strategy must be one of 'cos' or 'linear', "
                f"instead got {anneal_strategy}"
            )
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(
                f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}"
            )
        self.anneal_epochs = anneal_epochs
        self._set_anneal_func(anneal_strategy)
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    @override
    def get_lr(self):
        step = self.last_epoch
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [
            self._get_initial_lr(group["lr"], group["swa_lr"], prev_alpha)
            for group in self.optimizer.param_groups
        ]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [
            group["swa_lr"] * alpha + lr * (1 - alpha)
            for group, lr in zip(self.optimizer.param_groups, prev_lrs, strict=True)
        ]

    def _set_anneal_func(self, anneal_strategy: Literal["cos", "linear"]) -> None:
        self._anneal_strategy = anneal_strategy
        self.anneal_func = (
            self._cosine_anneal
            if anneal_strategy == "cos"
            else self._linear_anneal
        )

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "anneal_func")
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        self._set_anneal_func(self._anneal_strategy)
