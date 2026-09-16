"""Numerical comparison utilities for float and quantized models.

The tools in this module record intermediate tensors produced by a float
model and a quantized model that consume the same input, pair the recorded
tensors at matching module locations, and hand the pairs back to the caller
for error analysis.  Three comparison levels are covered:

- weights: :func:`compare_weights` pairs weight entries from two state
  dicts by module path.
- module outputs under shadowing: :func:`compare_model_stub` wraps
  quantized modules in :class:`Shadow` so a float module with the same
  input runs alongside and :class:`ShadowLogger` records both outputs.
- module outputs across models: :func:`compare_model_outputs` attaches
  :class:`OutputLogger` to matching modules of both models and pairs the
  recorded activations with :func:`get_matching_activations`.

Recorded statistics stay attached to the modules, so they can be collected
at any time with :func:`get_logger_dict`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorplay
from tensorplay import nn

__all__ = [
    "DEFAULT_COMPARE_OUTPUT_MODULE_LIST",
    "NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST",
    "Logger",
    "OutputLogger",
    "Shadow",
    "ShadowLogger",
    "compare_model_outputs",
    "compare_model_stub",
    "compare_weights",
    "get_logger_dict",
    "get_matching_activations",
    "prepare_model_outputs",
    "prepare_model_with_stubs",
]

# Quantized container modules whose outputs may carry a logger even though
# they are not simple leaf ops.  The float LSTM is listed as well because
# its internal structure needs the logger at the module boundary.  The
# quantized module library is optional; when it is unavailable the allow
# list degrades to the float module types only.
NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST: set[type] = {nn.LSTM}
try:
    from tensorplay.ao.nn.quantized.linear import QuantizedLinear as _QuantizedLinear

    NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST.add(_QuantizedLinear)
except ImportError:  # pragma: no cover - depends on optional kernels
    pass

# Module types whose outputs are recorded by :func:`compare_model_outputs`
# when the caller does not supply an explicit allow list: every float module
# that has a quantized counterpart, every quantized module produced by
# conversion, and plain containers.
from tensorplay.ao.quantization.quantization_mappings import (
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_QAT_MODULE_MAPPINGS,
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
)

DEFAULT_COMPARE_OUTPUT_MODULE_LIST: set[type] = (
    set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.keys())
    | set(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS.values())
    | set(DEFAULT_QAT_MODULE_MAPPINGS.keys())
    | set(DEFAULT_QAT_MODULE_MAPPINGS.values())
    | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.keys())
    | set(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS.values())
    | {nn.Sequential}
)


def _find_match(
    str_list: dict[str, Any] | list[str],
    key_str: str,
    postfix: str,
) -> str | None:
    """Find the entry of ``str_list`` that names the same module as
    ``key_str`` while ending with ``postfix``.

    ``key_str`` is a dotted name whose last component must equal
    ``postfix``; the remaining components form the module path to match.
    A candidate matches when its own module path (dropping its last one
    or two components) equals that path.

    Args:
        str_list: candidate names, for example the keys of a state dict
        key_str: dotted name whose last component is ``postfix``
        postfix: required last component of ``key_str``

    Return:
        The matching candidate, or ``None`` when nothing matches.
    """
    split_str = key_str.split(".")
    if split_str[-1] != postfix:
        return None
    match_string = "".join(key_str.split(".")[0:-1])
    for s2 in str_list:
        pattern1 = "".join(s2.split(".")[0:-1])
        pattern2 = "".join(s2.split(".")[0:-2])
        if match_string == pattern1:
            return s2
        if match_string == pattern2:
            return s2
    return None


def compare_weights(
    float_dict: dict[str, Any], quantized_dict: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Pair the weights of matching modules from two state dicts.

    Returns a dict keyed by the quantized state dict entry names; each
    entry holds the tensors under ``"float"`` and ``"quantized"``.  A
    quantized weight entry is paired with the float weight entry of the
    same module path, so a quantized buffer named ``fc.qweight`` pairs
    with the float ``fc.weight``.  Quantized entries are returned as
    stored; call ``dequantize()`` on them to compare against the float
    values.

    Example::

        wt_compare_dict = compare_weights(
            float_model.state_dict(), q_model.state_dict())
        for key, pair in wt_compare_dict.items():
            print(key, pair["float"], pair["quantized"])

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict keyed by quantized state dict entry names, with
        each entry holding ``"float"`` and ``"quantized"`` tensors
    """
    weight_dict: dict[str, dict[str, Any]] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is None:
            # Quantized modules store their weights as buffers whose names
            # carry a "q" prefix on the float weight attribute name.
            match_key = _find_match(float_dict, key, "qweight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
            continue

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        # (the packed entry is a (weight, bias) pair) or a flat buffer that
        # merely carries the "_packed_params" name.
        match_key = _find_match(float_dict, key, "_packed_params")
        if match_key is not None:
            entry = quantized_dict[key]
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = (
                entry[0] if isinstance(entry, (tuple, list)) else entry)
    return weight_dict


def _get_logger_dict_helper(
    mod: nn.Module,
    target_dict: dict[str, Any],
    prefix: str = "",
) -> None:
    """Collect the stats of every logger found below ``mod`` into
    ``target_dict``, keyed by ``"<module path>.stats"``.

    Args:
        mod: module whose subtree is traversed
        prefix: dotted path of ``mod`` within its root model
        target_dict: output dict receiving the stats entries
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    for child in mod.children():
        if isinstance(child, Logger):
            target_dict[get_prefix(prefix) + "stats"] = child.stats
            break

    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_logger_dict_helper(child, target_dict, module_prefix)


def get_logger_dict(mod: nn.Module, prefix: str = "") -> dict[str, dict]:
    """Traverse the module subtree of ``mod`` and collect the stats of all
    attached loggers.

    Logger types handled:

        ShadowLogger: records the outputs of a quantized module and its
        float shadow module,
        OutputLogger: records the outputs of the module it is attached to.

    Args:
        mod: module whose loggers are collected
        prefix: dotted path of ``mod`` within its root model

    Return:
        target_dict: dict mapping ``"<module path>.stats"`` to the stats
        dict of the corresponding logger
    """
    target_dict: dict[str, dict] = {}
    _get_logger_dict_helper(mod, target_dict, prefix)
    return target_dict


class Logger(nn.Module):
    """Base class for modules that collect statistics during forward."""

    def __init__(self):
        super().__init__()
        self.stats: dict[str, Any] = {}
        # Dtype of the activations this logger is prepared to record; only
        # static quantization produces unsigned 8-bit activations.
        self.dtype = tensorplay.quint8

    def forward(self, x):
        """Base loggers record nothing; subclasses override this."""


class ShadowLogger(Logger):
    """Records the outputs of the original module and of its float shadow
    module, keeping them under the ``"quantized"`` and ``"float"`` stats
    keys.
    """

    def __init__(self):
        super().__init__()
        self.stats["float"] = []
        self.stats["quantized"] = []

    def forward(self, x, y):
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats["quantized"].append(x.detach())
        self.stats["float"].append(y.detach())


class OutputLogger(Logger):
    """Records the output of the module it is attached to under the
    ``"tensor_val"`` stats key and leaves the output unchanged.
    """

    def __init__(self):
        super().__init__()
        self.stats["tensor_val"] = []

    def forward(self, x):
        self.stats["tensor_val"].append(x)
        return x


def _convert_tuple_to_list(t: Any) -> Any:
    return [_convert_tuple_to_list(x) for x in t] if type(t) is tuple else t


def _dequantize_tensor_list(t: Any) -> Any:
    return (
        [_dequantize_tensor_list(x) for x in t]
        if type(t) is list
        else t.dequantize()
        if t.is_quantized()
        else t
    )


class Shadow(nn.Module):
    """Runs a float module alongside the quantized module it replaces.

    The quantized module stays the only contributor to the model output;
    the float module receives the same (dequantized) input and both
    outputs are handed to ``logger_cls`` for recording.

    Args:
        q_module: quantized module producing the model output
        float_module: float module fed with the same input for reference
        logger_cls: logger type processing the two outputs; a
            :class:`ShadowLogger` records them for later comparison
    """

    def __init__(self, q_module, float_module, logger_cls):
        super().__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.logger = logger_cls()

    def forward(self, *x) -> Any:
        xl = _convert_tuple_to_list(x)
        output = self.orig_module(*xl)
        xl_float = _dequantize_tensor_list(xl)
        shadow_output = self.shadow_module(*xl_float)
        self.logger(output, shadow_output)
        return output


def prepare_model_with_stubs(
    float_module: nn.Module,
    q_module: nn.Module,
    module_swap_list: set[type],
    logger_cls: Callable,
) -> None:
    """Wrap the quantized modules of ``q_module`` in :class:`Shadow`.

    Recursion descends in lock step through both models.  Whenever the
    float module type is in ``module_swap_list`` and the corresponding
    quantized module has a different type, the quantized child is replaced
    in place by a :class:`Shadow` that keeps the original quantized module
    as the output producer and the float module as the shadow.

    Example::

        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model)

    Args:
        float_module: float model the quantized model was derived from
        q_module: quantized model, modified in place
        module_swap_list: float module types at which shadows are attached
        logger_cls: logger type used inside each :class:`Shadow`
    """
    float_module_children = dict(float_module.named_children())

    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)

        # Attach a shadow only when the quantized child is not simply the
        # same module type as its float counterpart.
        if type(float_mod) in module_swap_list and not _is_identical_module_type(
            mod, float_mod
        ):
            reassign[name] = Shadow(mod, float_mod, logger_cls)

    for key, value in reassign.items():
        q_module._modules[key] = value


def _is_identical_module_type(mod1, mod2):
    """Return whether both module subtrees are built from the same module
    types in the same order."""
    mod1_module_types = [type(mod) for mod in mod1.modules()]
    mod2_module_types = [type(mod) for mod in mod2.modules()]
    return mod1_module_types == mod2_module_types


def compare_model_stub(
    float_model: nn.Module,
    q_model: nn.Module,
    module_swap_list: set[type],
    *data,
    logger_cls=ShadowLogger,
) -> dict[str, dict]:
    """Compare a quantized module against its float counterpart on the
    same input.

    The models are prepared with :func:`prepare_model_with_stubs`, the
    quantized model runs on ``data``, and the recorded stats are collected
    with :func:`get_logger_dict`.  The result maps module paths to stats
    dicts holding ``"float"`` and ``"quantized"`` output tensors, which
    the caller can use to compute module-level quantization error.

    Example::

        module_swap_list = [nn.Linear]
        ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
        for key, pair in ob_dict.items():
            print(key, pair["float"], pair["quantized"])

    Args:
        float_model: float model the quantized model was derived from
        q_model: quantized model, modified in place
        module_swap_list: float module types at which shadows are attached
        data: input data run through the prepared quantized model
        logger_cls: logger type used inside each :class:`Shadow`

    Return:
        ob_dict: dict mapping module paths to the recorded stats
    """
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    q_model(*data)
    ob_dict = get_logger_dict(q_model)
    return ob_dict


def get_matching_activations(
    float_module: nn.Module,
    q_module: nn.Module,
) -> dict[str, dict[str, Any]]:
    """Pair the activations recorded in the float and the quantized model.

    Both models must have been prepared with :func:`prepare_model_outputs`
    and run on the same input beforehand.

    Args:
        float_module: float model with attached output loggers
        q_module: quantized model with attached output loggers

    Return:
        act_dict: dict keyed by quantized module paths; each entry holds
        the recorded activations under ``"float"`` and ``"quantized"``
    """
    float_dict = get_logger_dict(float_module)
    quantized_dict = get_logger_dict(q_module)
    act_dict: dict[str, dict] = {}
    for key in quantized_dict:
        if len(quantized_dict[key]["tensor_val"]) == 0:
            continue
        match_key = _find_match(sorted(float_dict, reverse=True), key, "stats")
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key]["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key]["tensor_val"]
    return act_dict


def _attach_output_logger(
    module: nn.Module, logger_cls: Callable[[], Logger]
) -> None:
    """Attach a fresh logger to ``module`` as a child module and register a
    forward hook that feeds the module output to the logger.

    Args:
        module: module whose outputs are recorded
        logger_cls: logger type instantiated for the module
    """
    logger = logger_cls()
    module.add_module("output_logger", logger)

    def _record_output(mod, inputs, output):
        logger(output)

    module.register_forward_hook(_record_output)


def _insert_output_loggers(
    mod: nn.Module,
    logger_cls: Callable[[], Logger],
    allow_list: set[type],
    non_leaf_allow_list: set[type],
) -> None:
    """Attach loggers to the subtree of ``mod``.

    A logger is attached to every child whose type is in ``allow_list``
    or in ``non_leaf_allow_list``; traversal skips existing loggers so a
    repeated preparation does not double-instrument a module.

    Args:
        mod: module whose subtree is instrumented
        logger_cls: logger type instantiated per instrumented module
        allow_list: module types that receive a logger
        non_leaf_allow_list: additional module types that receive a
            logger even though they are not simple leaf ops
    """
    for _, child in mod.named_children():
        if isinstance(child, Logger):
            continue
        if type(child) in allow_list or type(child) in non_leaf_allow_list:
            _attach_output_logger(child, logger_cls)
        _insert_output_loggers(child, logger_cls, allow_list, non_leaf_allow_list)


def prepare_model_outputs(
    float_module: nn.Module,
    q_module: nn.Module,
    logger_cls=OutputLogger,
    allow_list=None,
) -> None:
    """Attach output loggers to both models.

    Every module whose type is in ``allow_list`` receives a logger in the
    float model; the quantized model additionally honors
    ``NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST`` so quantized container
    modules are instrumented as well.

    Args:
        float_module: float model the quantized model was derived from
        q_module: quantized model
        logger_cls: logger type attached to instrumented modules
        allow_list: module types that receive a logger; defaults to
            ``DEFAULT_COMPARE_OUTPUT_MODULE_LIST``
    """
    if allow_list is None:
        allow_list = DEFAULT_COMPARE_OUTPUT_MODULE_LIST

    _insert_output_loggers(float_module, logger_cls, allow_list, set())
    _insert_output_loggers(
        q_module,
        logger_cls,
        allow_list,
        NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,
    )


def compare_model_outputs(
    float_model: nn.Module,
    q_model: nn.Module,
    *data,
    logger_cls=OutputLogger,
    allow_list=None,
) -> dict[str, dict[str, Any]]:
    """Compare the outputs of matching modules between the float and the
    quantized model on the same input.

    Both models are instrumented with :func:`prepare_model_outputs`, run
    on ``data``, and the recorded activations are paired with
    :func:`get_matching_activations`.  The result maps quantized module
    paths to dicts holding ``"float"`` and ``"quantized"`` activation
    lists, which the caller can use to trace how quantization error
    propagates through the model.

    Example::

        act_compare_dict = compare_model_outputs(float_model, q_model, data)
        for key, pair in act_compare_dict.items():
            print(key, pair["float"], pair["quantized"])

    Args:
        float_model: float model the quantized model was derived from
        q_model: quantized model
        data: input data run through both models
        logger_cls: logger type attached to instrumented modules
        allow_list: module types that receive a logger; defaults to
            ``DEFAULT_COMPARE_OUTPUT_MODULE_LIST``

    Return:
        act_compare_dict: dict keyed by quantized module paths; each entry
        holds the recorded activations under ``"float"`` and
        ``"quantized"``
    """
    if allow_list is None:
        allow_list = DEFAULT_COMPARE_OUTPUT_MODULE_LIST
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)
    float_model(*data)
    q_model(*data)
    act_compare_dict = get_matching_activations(float_model, q_model)
    return act_compare_dict
