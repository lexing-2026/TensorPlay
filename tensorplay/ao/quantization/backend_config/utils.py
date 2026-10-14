"""Helpers for translating backend configs into internal pattern formats."""

from collections.abc import Callable

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from ..fuser_method_mappings import _reverse2, _reverse3
from ..utils import Pattern

from .backend_config import BackendConfig, BackendPatternConfig, DTypeConfig

__all__ = [
    "get_pattern_to_dtype_configs",
    "get_qat_module_classes",
    "get_fused_module_classes",
    "get_pattern_to_input_type_to_index",
    "get_root_module_to_quantized_reference_module",
    "get_fuser_method_mapping",
    "get_module_to_qat_module",
    "get_fusion_pattern_to_root_node_getter",
    "get_fusion_pattern_to_extra_inputs_getter",
    "remove_boolean_dispatch_from_name",
    "pattern_to_human_readable",
    "entry_to_pretty_str",
]


def get_pattern_to_dtype_configs(backend_config: BackendConfig) -> dict:
    return {
        pattern: config.dtype_configs
        for pattern, config in backend_config._pattern_complex_format_to_config.items()
    }


def get_qat_module_classes(backend_config: BackendConfig) -> tuple:
    qat_module_classes = [
        config.qat_module
        for config in backend_config.configs
        if config.qat_module is not None
    ]
    return tuple(set(qat_module_classes))


def get_fused_module_classes(backend_config: BackendConfig) -> tuple:
    fused_module_classes = [
        config.fused_module
        for config in backend_config.configs
        if config.fused_module is not None
    ]
    return tuple(set(fused_module_classes))


def get_pattern_to_input_type_to_index(backend_config: BackendConfig) -> dict:
    return {
        pattern: config._input_type_to_index
        for pattern, config in backend_config._pattern_complex_format_to_config.items()
    }


def get_root_module_to_quantized_reference_module(
    backend_config: BackendConfig,
) -> dict:
    mapping = {}
    for config in backend_config.configs:
        if config.root_module is not None and config.reference_quantized_module is not None:
            mapping[config.root_module] = config.reference_quantized_module
    return mapping


def _get_fuser_method_in_reversed_nested_tuple_format(
    config: BackendPatternConfig,
) -> Callable:
    if config.fuser_method is None:
        raise AssertionError("config.fuser_method must be provided")
    if config._pattern_complex_format is not None:
        return config.fuser_method
    if not isinstance(config.pattern, tuple):
        raise ValueError(f"Expected pattern to be a tuple, got: {config.pattern}")
    if len(config.pattern) == 2:
        return _reverse2(config.fuser_method)
    elif len(config.pattern) == 3:
        return _reverse3(config.fuser_method)
    raise ValueError(f"Expected a tuple with 2 or 3 elements, got: {config.pattern}")


def get_fuser_method_mapping(backend_config: BackendConfig) -> dict:
    fuser_method_mapping = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            fuser_method_mapping[pattern] = _get_fuser_method_in_reversed_nested_tuple_format(config)
    return fuser_method_mapping


def get_module_to_qat_module(backend_config: BackendConfig) -> dict:
    return {
        pattern: config.qat_module
        for pattern, config in backend_config._pattern_complex_format_to_config.items()
        if config.qat_module is not None
    }


def get_fusion_pattern_to_root_node_getter(backend_config: BackendConfig) -> dict:
    return {
        pattern: config._root_node_getter
        for pattern, config in backend_config._pattern_complex_format_to_config.items()
        if config._root_node_getter is not None
    }


def get_fusion_pattern_to_extra_inputs_getter(backend_config: BackendConfig) -> dict:
    return {
        pattern: config._extra_inputs_getter
        for pattern, config in backend_config._pattern_complex_format_to_config.items()
        if config._extra_inputs_getter is not None
    }


def remove_boolean_dispatch_from_name(p):
    if p is F.fractional_max_pool2d:
        return "tensorplay.nn.functional.fractional_max_pool2d"
    elif p is F.fractional_max_pool3d:
        return "tensorplay.nn.functional.fractional_max_pool3d"
    elif p is F.max_pool1d:
        return "tensorplay.nn.functional.max_pool1d"
    elif p is F.max_pool2d:
        return "tensorplay.nn.functional.max_pool2d"
    elif p is F.max_pool3d:
        return "tensorplay.nn.functional.max_pool3d"
    elif p is F.adaptive_max_pool1d:
        return "tensorplay.nn.functional.adaptive_max_pool1d"
    elif p is F.adaptive_max_pool2d:
        return "tensorplay.nn.functional.adaptive_max_pool2d"
    elif p is F.adaptive_max_pool3d:
        return "tensorplay.nn.functional.adaptive_max_pool3d"
    return p


def pattern_to_human_readable(p):
    if isinstance(p, tuple):
        return tuple(pattern_to_human_readable(inner_p) for inner_p in p)
    elif isinstance(p, str):
        return p
    else:
        return remove_boolean_dispatch_from_name(p)


def entry_to_pretty_str(entry) -> str:
    s = "{\n"
    if "pattern" in entry:
        s += f"  'pattern': {pattern_to_human_readable(entry['pattern'])},\n"
    if "dtype_configs" in entry:
        s += "  'dtype_configs': [\n"
        for dtype_config in entry["dtype_configs"]:
            s += "    {\n"
            for k, v in dtype_config.items():
                s += f"      '{k}': {v},\n"
            s += "    },\n"
        s += "  ],\n"
    if "num_tensor_args_to_observation_type" in entry:
        s += "  'num_tensor_args_to_observation_type': {\n"
        for k, v in entry["num_tensor_args_to_observation_type"].items():
            s += f"    {k}: {v},\n"
        s += "  },\n"
    custom_handled_fields = ["pattern", "dtype_configs", "num_tensor_args_to_observation_type"]
    for field_name in entry:
        if field_name in custom_handled_fields:
            continue
        s += f"  '{field_name}': {entry[field_name]},\n"
    s += "}"
    return s


def _get_pattern_in_reversed_nested_tuple_format(
    config: BackendPatternConfig,
) -> Pattern:
    """Convert a user-facing pattern to the internal reversed nested tuple form."""
    if config._pattern_complex_format is not None:
        return config._pattern_complex_format
    if config.pattern is None:
        raise ValueError("Either 'pattern' or 'pattern_complex_format' must be specified")
    if not isinstance(config.pattern, tuple):
        return config.pattern
    if len(config.pattern) == 2:
        (a, b) = config.pattern
        return (b, a)
    elif len(config.pattern) == 3:
        (a, b, c) = config.pattern
        return (c, (b, a))
    raise ValueError(f"Expected a tuple with 2 or 3 elements, got: {config.pattern}")