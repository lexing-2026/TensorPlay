# Ported from torch/distributed/algorithms/model_averaging/utils.py.
import itertools
from collections.abc import Iterable, Iterator

import tensorplay as tp
import tensorplay.distributed as dist

from tensorplay.distributed import GroupMember, ProcessGroup


__all__ = [
    "average_parameters",
    "get_params_to_average",
    "average_parameters_or_parameter_groups",
]


def average_parameters(
    params: Iterator[tp.nn.Parameter], process_group: ProcessGroup
):
    """
    Averages all the given parameters.

    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    group_to_use = process_group if process_group is not None else GroupMember.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    params_it1, params_it2 = itertools.tee(params)
    # If the input parameters have different data types,
    # packing these parameters will trigger an implicit type up-casting.
    # The original parameter data types will be restored during the subsequent unpacking.
    flat_params = tp.cat([p.data.reshape(-1) for p in params_it1])
    flat_params /= dist.get_world_size(group_to_use)
    dist.all_reduce(flat_params, group=group_to_use)

    offset = 0
    for p in params_it2:
        p.data = flat_params[offset : offset + p.numel()].view(p.shape).to(p.dtype)
        offset += p.numel()


def get_params_to_average(
    params: Iterable[tp.nn.Parameter] | Iterable[dict[str, tp.nn.Parameter]],
):
    """
    Return a list of parameters that need to average.

    This filters out the parameters that do not contain any gradients.
    Args:
        params: The parameters of a model or parameter groups of an optimizer.
    """
    filtered_params = []
    for param in params:
        if isinstance(param, tp.nn.Parameter):
            # model.parameters() input
            param_data = param
            if param_data.grad is not None:
                filtered_params.append(param_data)
        elif isinstance(param, dict):
            # optimizer.param_groups input
            for param_data in param["params"]:
                if param_data.grad is not None:
                    filtered_params.append(param_data)
        else:
            raise NotImplementedError(
                f"Parameter input of type {type(param)} is not supported"
            )
    return filtered_params


def average_parameters_or_parameter_groups(
    params: Iterable[tp.nn.Parameter] | Iterable[dict[str, tp.nn.Parameter]],
    process_group: ProcessGroup,
):
    """Averages parameters of a model or parameter groups of an optimizer."""
    average_parameters(iter(get_params_to_average(params)), process_group)
