# mypy: allow-untyped-defs
#
# runs inline when the bucket's gradient communication completes.
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, no_type_check

import tensorplay as tp

import tensorplay.distributed as dist


__all__: list[str] = []

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"


class _OptimizerHookState:
    """
    Holds state for running optimizer in-line after DDP communication hook.

    Currently contains only optimizer class which must have a method `step_param`.
    """

    __slots__ = ["functional_optimizer", "params_to_optimize"]

    def __init__(self, functional_optim, params=None):
        self.functional_optimizer = functional_optim
        self._check_valid_functional_optim()
        self._set_params_to_optimize(params)

    def _set_params_to_optimize(self, params):
        if params is not None:
            self.params_to_optimize = set(params)

    def _check_valid_functional_optim(self):
        if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
            raise ValueError(
                f"Class {type(self.functional_optimizer)} must implement method "
                f"{_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}."
            )


@dataclass
class _OptimInBackwardHookState:
    # tp has no side-stream machinery; retain the public state field.
    wait_for_optim_stream_enqueued: bool = False


@no_type_check
def _apply_optim_in_backward_hook(
    gradient_is_bucket_view: bool,
) -> Callable[[Any, dist.GradBucket], Any]:
    r"""
    Register hook to apply the optimizer in backward.

    If tensorplay.distributed.optim._apply_optimizer_in_backward is used to overlap
    optimizer with backward pass, DDP will run the below hook to run optimizer
    step for parameters after gradient communication has taken place.
    """
    optim_in_bwd_state = _OptimInBackwardHookState()

    def _hook(state: _OptimizerHookState, bucket: dist.GradBucket):
        return _run_optim_in_backward_hook(
            state,
            bucket,
            optim_in_bwd_state=optim_in_bwd_state,
            gradient_is_bucket_view=gradient_is_bucket_view,
        )

    return _hook


def _run_optim_in_backward_hook(
    optim_hook_state: _OptimizerHookState,
    bucket: dist.GradBucket,
    optim_in_bwd_state: _OptimInBackwardHookState,
    gradient_is_bucket_view: bool,
):
    # Run optimizer step for all params that have requested optim-in-backward.
    for param in bucket.parameters():
        has_attr = hasattr(param, "_in_backward_optimizers")
        if has_attr:
            for optimizer in param._in_backward_optimizers:
                optimizer.step_param(param, param.grad)

    # Remove any params that have been optimized by the above hook.
    params = []
    for param in bucket.parameters():
        if not hasattr(param, "_in_backward_optimizers"):
            params.append(param)
    if params:
        optim_hook_state.functional_optimizer.step_param(
            params[0], params[0].grad
        )
