"""Backend quantization capability descriptors.

``BackendConfig`` declares which operator patterns a backend can quantize,
which dtypes they accept, how they are observed, and how they fuse and lower
to reference quantized modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import tensorplay as tp

__all__ = [
    "BackendConfig",
    "BackendPatternConfig",
    "DTypeConfig",
    "DTypeWithConstraints",
    "ObservationType",
]

INPUT_DTYPE_DICT_KEY = "input_dtype"
OUTPUT_DTYPE_DICT_KEY = "output_dtype"
WEIGHT_DTYPE_DICT_KEY = "weight_dtype"
BIAS_DTYPE_DICT_KEY = "bias_dtype"
IS_DYNAMIC_DICT_KEY = "is_dynamic"

NAME_DICT_KEY = "name"
CONFIGS_DICT_KEY = "configs"

PATTERN_DICT_KEY = "pattern"
PATTERN_COMPLEX_FORMAT_DICT_KEY = "pattern_complex_format"
OBSERVATION_TYPE_DICT_KEY = "observation_type"
DTYPE_CONFIGS_DICT_KEY = "dtype_configs"
ROOT_MODULE_DICT_KEY = "root_module"
QAT_MODULE_DICT_KEY = "qat_module"
REFERENCE_QUANTIZED_MODULE_DICT_KEY = "reference_quantized_module_for_root"
FUSED_MODULE_DICT_KEY = "fused_module"
FUSER_METHOD_DICT_KEY = "fuser_method"
ROOT_NODE_GETTER_DICT_KEY = "root_node_getter"
EXTRA_INPUTS_GETTER_DICT_KEY = "extra_inputs_getter"
NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY = "num_tensor_args_to_observation_type"
INPUT_TYPE_TO_INDEX_DICT_KEY = "input_type_to_index"


class ObservationType(Enum):
    """How an operator pattern should be observed."""

    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    INPUT_OUTPUT_NOT_OBSERVED = 2


@dataclass
class DTypeWithConstraints:
    """A dtype plus optional quantization-range and scale constraints."""

    dtype: tp.dtype | None = None
    quant_min_lower_bound: int | float | None = None
    quant_max_upper_bound: int | float | None = None
    scale_min_lower_bound: int | float | None = None
    scale_max_upper_bound: int | float | None = None
    scale_exact_match: float | None = None
    zero_point_exact_match: int | None = None


@dataclass
class DTypeConfig:
    """Supported dtypes for a pattern's input, output, weight and bias.

    These dtypes are matched against the user's qconfig; a match means the
    pattern can be quantized with that qconfig.
    """

    input_dtype_with_constraints: DTypeWithConstraints
    output_dtype_with_constraints: DTypeWithConstraints
    weight_dtype_with_constraints: DTypeWithConstraints
    bias_dtype: tp.dtype | None
    is_dynamic: bool | None

    def __init__(
        self,
        input_dtype: tp.dtype | DTypeWithConstraints | None = None,
        output_dtype: tp.dtype | DTypeWithConstraints | None = None,
        weight_dtype: tp.dtype | DTypeWithConstraints | None = None,
        bias_dtype: tp.dtype | None = None,
        is_dynamic: bool | None = None,
    ):
        if isinstance(input_dtype, DTypeWithConstraints):
            self.input_dtype_with_constraints = input_dtype
        else:
            self.input_dtype_with_constraints = DTypeWithConstraints(dtype=input_dtype)

        if isinstance(output_dtype, DTypeWithConstraints):
            self.output_dtype_with_constraints = output_dtype
        else:
            self.output_dtype_with_constraints = DTypeWithConstraints(dtype=output_dtype)

        if isinstance(weight_dtype, DTypeWithConstraints):
            self.weight_dtype_with_constraints = weight_dtype
        else:
            self.weight_dtype_with_constraints = DTypeWithConstraints(dtype=weight_dtype)

        self.bias_dtype = bias_dtype
        self.is_dynamic = is_dynamic

    @property
    def input_dtype(self):
        return self.input_dtype_with_constraints.dtype

    @property
    def output_dtype(self):
        return self.output_dtype_with_constraints.dtype

    @property
    def weight_dtype(self):
        return self.weight_dtype_with_constraints.dtype

    @classmethod
    def from_dict(cls, dtype_config_dict: dict[str, Any]) -> "DTypeConfig":
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY)
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY)
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY)
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY)
        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY)
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> dict[str, Any]:
        dtype_config_dict: dict[str, Any] = {}
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype_with_constraints
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = self.output_dtype_with_constraints
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = self.weight_dtype_with_constraints
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        return dtype_config_dict


class BackendPatternConfig:
    """Configuration for one quantizable operator pattern."""

    def __init__(self, pattern=None):
        self.pattern = pattern
        self._pattern_complex_format = None
        self._observation_type = None
        self._dtype_configs = []
        self._root_module = None
        self._qat_module = None
        self._reference_quantized_module = None
        self._fused_module = None
        self._fuser_method = None
        self._root_node_getter = None
        self._extra_inputs_getter = None
        self._num_tensor_args_to_observation_type = {}
        self._input_type_to_index = {}

    def set_pattern(self, pattern):
        self.pattern = pattern
        return self

    def set_pattern_complex_format(self, pattern_complex_format):
        self._pattern_complex_format = pattern_complex_format
        return self

    def set_observation_type(self, observation_type):
        self._observation_type = observation_type
        return self

    def add_dtype_config(self, dtype_config):
        self._dtype_configs.append(dtype_config)
        return self

    def set_dtype_configs(self, dtype_configs):
        self._dtype_configs = list(dtype_configs)
        return self

    def set_root_module(self, root_module):
        self._root_module = root_module
        return self

    def set_qat_module(self, qat_module):
        self._qat_module = qat_module
        return self

    def set_reference_quantized_module(self, reference_quantized_module):
        self._reference_quantized_module = reference_quantized_module
        return self

    def set_fused_module(self, fused_module):
        self._fused_module = fused_module
        return self

    def set_fuser_method(self, fuser_method):
        self._fuser_method = fuser_method
        return self

    def set_root_node_getter(self, root_node_getter):
        self._root_node_getter = root_node_getter
        return self

    def set_extra_inputs_getter(self, extra_inputs_getter):
        self._extra_inputs_getter = extra_inputs_getter
        return self

    def set_num_tensor_args_to_observation_type(self, mapping):
        self._num_tensor_args_to_observation_type = mapping
        return self

    def set_input_type_to_index(self, mapping):
        self._input_type_to_index = mapping
        return self

    @property
    def observation_type(self):
        return self._observation_type

    @property
    def dtype_configs(self):
        return list(self._dtype_configs)

    @property
    def root_module(self):
        return self._root_module

    @property
    def qat_module(self):
        return self._qat_module

    @property
    def reference_quantized_module(self):
        return self._reference_quantized_module

    @property
    def fused_module(self):
        return self._fused_module

    @property
    def fuser_method(self):
        return self._fuser_method

    @property
    def num_tensor_args_to_observation_type(self):
        return dict(self._num_tensor_args_to_observation_type)

    @property
    def input_type_to_index(self):
        return dict(self._input_type_to_index)

    @classmethod
    def from_dict(cls, backend_pattern_config_dict: dict[str, Any]) -> "BackendPatternConfig":
        config = cls()
        if PATTERN_DICT_KEY in backend_pattern_config_dict:
            config.set_pattern(backend_pattern_config_dict[PATTERN_DICT_KEY])
        if PATTERN_COMPLEX_FORMAT_DICT_KEY in backend_pattern_config_dict:
            config.set_pattern_complex_format(
                backend_pattern_config_dict[PATTERN_COMPLEX_FORMAT_DICT_KEY]
            )
        if OBSERVATION_TYPE_DICT_KEY in backend_pattern_config_dict:
            config.set_observation_type(
                backend_pattern_config_dict[OBSERVATION_TYPE_DICT_KEY]
            )
        for d in backend_pattern_config_dict.get(DTYPE_CONFIGS_DICT_KEY, []):
            config.add_dtype_config(
                DTypeConfig.from_dict(d) if isinstance(d, dict) else d
            )
        if ROOT_MODULE_DICT_KEY in backend_pattern_config_dict:
            config.set_root_module(backend_pattern_config_dict[ROOT_MODULE_DICT_KEY])
        if QAT_MODULE_DICT_KEY in backend_pattern_config_dict:
            config.set_qat_module(backend_pattern_config_dict[QAT_MODULE_DICT_KEY])
        if REFERENCE_QUANTIZED_MODULE_DICT_KEY in backend_pattern_config_dict:
            config.set_reference_quantized_module(
                backend_pattern_config_dict[REFERENCE_QUANTIZED_MODULE_DICT_KEY]
            )
        if FUSED_MODULE_DICT_KEY in backend_pattern_config_dict:
            config.set_fused_module(backend_pattern_config_dict[FUSED_MODULE_DICT_KEY])
        if FUSER_METHOD_DICT_KEY in backend_pattern_config_dict:
            config.set_fuser_method(backend_pattern_config_dict[FUSER_METHOD_DICT_KEY])
        if ROOT_NODE_GETTER_DICT_KEY in backend_pattern_config_dict:
            config.set_root_node_getter(
                backend_pattern_config_dict[ROOT_NODE_GETTER_DICT_KEY]
            )
        if EXTRA_INPUTS_GETTER_DICT_KEY in backend_pattern_config_dict:
            config.set_extra_inputs_getter(
                backend_pattern_config_dict[EXTRA_INPUTS_GETTER_DICT_KEY]
            )
        if NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY in backend_pattern_config_dict:
            config.set_num_tensor_args_to_observation_type(
                backend_pattern_config_dict[NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY]
            )
        if INPUT_TYPE_TO_INDEX_DICT_KEY in backend_pattern_config_dict:
            config.set_input_type_to_index(
                backend_pattern_config_dict[INPUT_TYPE_TO_INDEX_DICT_KEY]
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.pattern is not None:
            d[PATTERN_DICT_KEY] = self.pattern
        if self._pattern_complex_format is not None:
            d[PATTERN_COMPLEX_FORMAT_DICT_KEY] = self._pattern_complex_format
        if self._observation_type is not None:
            d[OBSERVATION_TYPE_DICT_KEY] = self._observation_type
        if self._dtype_configs:
            d[DTYPE_CONFIGS_DICT_KEY] = [c.to_dict() for c in self._dtype_configs]
        if self._root_module is not None:
            d[ROOT_MODULE_DICT_KEY] = self._root_module
        if self._qat_module is not None:
            d[QAT_MODULE_DICT_KEY] = self._qat_module
        if self._reference_quantized_module is not None:
            d[REFERENCE_QUANTIZED_MODULE_DICT_KEY] = self._reference_quantized_module
        if self._fused_module is not None:
            d[FUSED_MODULE_DICT_KEY] = self._fused_module
        if self._fuser_method is not None:
            d[FUSER_METHOD_DICT_KEY] = self._fuser_method
        if self._root_node_getter is not None:
            d[ROOT_NODE_GETTER_DICT_KEY] = self._root_node_getter
        if self._extra_inputs_getter is not None:
            d[EXTRA_INPUTS_GETTER_DICT_KEY] = self._extra_inputs_getter
        if self._num_tensor_args_to_observation_type:
            d[NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY] = self._num_tensor_args_to_observation_type
        if self._input_type_to_index:
            d[INPUT_TYPE_TO_INDEX_DICT_KEY] = self._input_type_to_index
        return d


class BackendConfig:
    """The set of patterns a backend can quantize."""

    def __init__(self, name: str = ""):
        self.name = name
        self._pattern_complex_format_to_config: dict = {}

    def __repr__(self):
        return f"BackendConfig({self.__dict__})"

    def set_name(self, name: str) -> "BackendConfig":
        self.name = name
        return self

    def set_backend_pattern_config(self, config: BackendPatternConfig) -> "BackendConfig":
        from .utils import _get_pattern_in_reversed_nested_tuple_format

        pattern_complex_format = _get_pattern_in_reversed_nested_tuple_format(config)
        self._pattern_complex_format_to_config[pattern_complex_format] = config
        return self

    def set_backend_pattern_configs(
        self, configs: list[BackendPatternConfig]
    ) -> "BackendConfig":
        for conf in configs:
            self.set_backend_pattern_config(conf)
        return self

    @property
    def configs(self) -> list[BackendPatternConfig]:
        return list(self._pattern_complex_format_to_config.values())

    @classmethod
    def from_dict(cls, backend_config_dict: dict[str, Any]) -> "BackendConfig":
        conf = cls(backend_config_dict.get(NAME_DICT_KEY, ""))
        for d in backend_config_dict.get(CONFIGS_DICT_KEY, []):
            if isinstance(d, BackendPatternConfig):
                conf.set_backend_pattern_config(d)
            elif isinstance(d, dict):
                conf.set_backend_pattern_config(BackendPatternConfig.from_dict(d))
        return conf

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {NAME_DICT_KEY: self.name}
        d[CONFIGS_DICT_KEY] = [config.to_dict() for config in self.configs]
        return d