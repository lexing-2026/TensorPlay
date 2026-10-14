"""Cross-layer weight equalization."""

import copy
from itertools import chain
from typing import Any

import tensorplay as tp
import tensorplay.ao.nn.intrinsic
from tensorplay import nn

__all__ = [
    "set_module_weight",
    "set_module_bias",
    "has_bias",
    "get_module_weight",
    "get_module_bias",
    "max_over_ndim",
    "min_over_ndim",
    "channel_range",
    "get_name_by_module",
    "cross_layer_equalization",
    "process_paired_modules_list_to_name",
    "expand_groups_in_paired_modules_list",
    "equalize",
    "converged",
]

_supported_types = {nn.Conv2d, nn.Linear, nn.Conv1d}
_supported_intrinsic_types = {
    tensorplay.ao.nn.intrinsic.ConvReLU2d,
    tensorplay.ao.nn.intrinsic.LinearReLU,
    tensorplay.ao.nn.intrinsic.ConvReLU1d,
}
_all_supported_types = _supported_types.union(_supported_intrinsic_types)


def set_module_weight(module, weight) -> None:
    if type(module) in _supported_types:
        module.weight = nn.Parameter(weight)
    else:
        module[0].weight = nn.Parameter(weight)


def set_module_bias(module, bias) -> None:
    if type(module) in _supported_types:
        module.bias = nn.Parameter(bias)
    else:
        module[0].bias = nn.Parameter(bias)


def has_bias(module) -> bool:
    if type(module) in _supported_types:
        return module.bias is not None
    return module[0].bias is not None


def get_module_weight(module):
    if type(module) in _supported_types:
        return module.weight
    return module[0].weight


def get_module_bias(module):
    if type(module) in _supported_types:
        return module.bias
    return module[0].bias


def max_over_ndim(input, axis_list, keepdim=False):
    """Apply max over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input


def min_over_ndim(input, axis_list, keepdim=False):
    """Apply min over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)
    return input


def channel_range(input, axis=0):
    """Find the per-channel range of a weight tensor."""
    size_of_tensor_dim = input.ndim
    axis_list = list(range(size_of_tensor_dim))
    axis_list.remove(axis)

    mins = min_over_ndim(input, axis_list)
    maxs = max_over_ndim(input, axis_list)

    if mins.size(0) != input.size(axis):
        raise AssertionError(
            "Dimensions of resultant channel range does not match size of requested axis"
        )
    return maxs - mins


def get_name_by_module(model, module):
    """Return the fully qualified name of ``module`` inside ``model``."""
    for name, m in model.named_modules():
        if m is module:
            return name
    raise ValueError("module is not in the model")


def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    """Scale the ranges so module1's output channels match module2's inputs."""
    if type(module1) not in _all_supported_types or type(module2) not in _all_supported_types:
        raise ValueError("module type not supported:", type(module1), " ", type(module2))

    bias = get_module_bias(module1) if has_bias(module1) else None

    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)

    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError(
            "Number of output channels of first arg do not match \
        number input channels of second arg"
        )

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    weight2_range += 1e-9
    scaling_factors = tp.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = tp.reciprocal(scaling_factors)

    if bias is not None:
        bias = bias * inverse_scaling_factors

    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = tp.reshape(scaling_factors, size2)
    inverse_scaling_factors = tp.reshape(inverse_scaling_factors, size1)

    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    set_module_weight(module1, weight1)
    if bias is not None:
        set_module_bias(module1, bias)
    set_module_weight(module2, weight2)


def process_paired_modules_list_to_name(model, paired_modules_list):
    """Convert module objects in the paired list to their names."""
    for group in paired_modules_list:
        for i, item in enumerate(group):
            if isinstance(item, nn.Module):
                group[i] = get_name_by_module(model, item)
            elif not isinstance(item, str):
                raise TypeError("item must be a nn.Module or a string")
    return paired_modules_list


def expand_groups_in_paired_modules_list(paired_modules_list):
    """Expand groups larger than two into adjacent pairs."""
    new_list = []
    for group in paired_modules_list:
        if len(group) == 1:
            raise ValueError("Group must have at least two modules")
        elif len(group) == 2:
            new_list.append(group)
        elif len(group) > 2:
            new_list.extend([group[i], group[i + 1]] for i in range(len(group) - 1))
    return new_list


def equalize(model, paired_modules_list, threshold=1e-4, inplace=True):
    """Iteratively equalize adjacent module pairs until convergence."""
    paired_modules_list = process_paired_modules_list_to_name(model, paired_modules_list)
    if not inplace:
        model = copy.deepcopy(model)
    paired_modules_list = expand_groups_in_paired_modules_list(paired_modules_list)

    name_to_module: dict[str, nn.Module] = {}
    previous_name_to_module: dict[str, Any] = {}
    name_set = set(chain.from_iterable(paired_modules_list))

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    while not converged(name_to_module, previous_name_to_module, threshold):
        for pair in paired_modules_list:
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])
            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])
    return model


def converged(curr_modules, prev_modules, threshold=1e-4):
    """Return whether the modules changed less than ``threshold``."""
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError(
            "The keys to the given mappings must have the same set of names of modules"
        )

    summed_norms = tp.tensor(0.0)
    if None in prev_modules.values():
        return False
    for name in curr_modules:
        curr_weight = get_module_weight(curr_modules[name])
        prev_weight = get_module_weight(prev_modules[name])
        difference = curr_weight.sub(prev_weight)
        summed_norms += tp.linalg.vector_norm(difference)
    return bool(summed_norms < threshold)