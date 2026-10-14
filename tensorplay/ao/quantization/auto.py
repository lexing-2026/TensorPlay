"""One-call model quantization workflows.

Convenience entry points that assign default qconfigs to every statically
quantizable layer, calibrate, and convert — a single-API alternative to the
manual prepare/convert dance.  This module is a convenience layer above the
eager workflow and does not require tracing the model.
"""

from __future__ import annotations

import copy
from typing import Callable

from tensorplay import nn

from .qconfig import default_per_channel_qconfig, default_qconfig
from .quantize import convert, prepare
from .quantization_mappings import get_default_static_quant_module_mappings

__all__ = ["auto_prepare", "auto_convert", "auto_quantize"]


def _assign_default_qconfigs(model, qconfig=None) -> None:
    """Assign a qconfig to every statically quantizable leaf without one.

    Conv layers get the per-channel default for better accuracy; other
    quantizable layers get the per-tensor default.
    """
    static_mappings = get_default_static_quant_module_mappings()
    for name, module in model.named_modules():
        if not name:
            continue
        if type(module) not in static_mappings:
            continue
        if getattr(module, "qconfig", None) is not None:
            continue
        if "Conv" in type(module).__name__:
            module.qconfig = qconfig or default_per_channel_qconfig
        else:
            module.qconfig = qconfig or default_qconfig


def auto_prepare(
    model: nn.Module,
    calibration_fn: Callable | None = None,
    qconfig=None,
    inplace: bool = False,
) -> nn.Module:
    """Assign default qconfigs and prepare ``model`` for static quantization.

    Args:
        model: the float model to prepare.
        calibration_fn: optional callable invoked with the prepared model; use
            it to run calibration data through the observers.
        qconfig: optional qconfig to apply to every quantizable layer (defaults
            to per-tensor/per-channel defaults depending on layer type).
        inplace: carry out preparation in-place.

    Returns:
        The prepared model (with observers attached).
    """
    if not inplace:
        model = copy.deepcopy(model)
    _assign_default_qconfigs(model, qconfig)
    prepare(model, inplace=True)
    if calibration_fn is not None:
        calibration_fn(model)
    return model


def auto_convert(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Convert an already-prepared model to its quantized counterpart."""
    if not inplace:
        model = copy.deepcopy(model)
    return convert(model, inplace=True)


def auto_quantize(
    model: nn.Module,
    calibration_fn: Callable | None = None,
    qconfig=None,
    inplace: bool = False,
) -> nn.Module:
    """Quantize ``model`` in one call: assign qconfigs, calibrate, convert.

    Args:
        model: the float model to quantize (should be in eval mode).
        calibration_fn: optional callable invoked with the prepared model to
            run representative calibration data.
        qconfig: optional qconfig override for every quantizable layer.
        inplace: carry out all transformations in-place.

    Returns:
        The statically quantized model, ready for inference.
    """
    if not inplace:
        model = copy.deepcopy(model)
    _assign_default_qconfigs(model, qconfig)
    prepare(model, inplace=True)
    if calibration_fn is not None:
        calibration_fn(model)
    return convert(model, inplace=True)