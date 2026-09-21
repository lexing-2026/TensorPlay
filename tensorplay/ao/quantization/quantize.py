"""Eager-mode quantization workflow.

``prepare`` attaches observers to float modules according to their
``qconfig`` attributes and records activation qparams through forward hooks
(each quantizable leaf gets an input-side and an output-side observer, since
the framework's quantized kernels take qparams as explicit arguments).
``convert`` swaps calibrated float modules for their quantized counterparts
via each target class's ``from_float``.  ``quantize`` chains the two around a
calibration callable; ``quantize_dynamic`` swaps modules for their
dynamic-quantization counterparts without calibration.
"""

from __future__ import annotations

import copy

from tensorplay import nn

from .fake_quantize import FakeQuantize
from .observer import ObserverBase
from .quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_static_quant_module_mappings,
)
from .stubs import DeQuantStub, QuantStub

__all__ = [
    "prepare",
    "convert",
    "quantize",
    "quantize_dynamic",
    "propagate_qconfig_",
    "add_quant_dequant",
]


def _matches(module, spec) -> bool:
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return isinstance(module, spec)
    if callable(spec):
        return spec(module)
    return False


def propagate_qconfig_(module, qconfig_dict=None):
    """Attach ``qconfig`` attributes derived from ``qconfig_dict``.

    Resolution per submodule: registered name first, then module type (by
    name or class), then the parent's qconfig.  A qconfig already set on the
    child wins over the inherited one.
    """
    qconfig_dict = qconfig_dict or {}
    for name, child in module.named_children():
        child.qconfig = (
            qconfig_dict.get(name)
            or qconfig_dict.get(type(child).__name__)
            or qconfig_dict.get(type(child))
            or getattr(child, "qconfig", None)
            or getattr(module, "qconfig", None)
        )
        propagate_qconfig_(child, None)


# ---------------------------------------------------------------------------
# observer attachment
# ---------------------------------------------------------------------------


def _observer_forward_hook(module, input, output):
    module.activation_post_process(output)
    return output


def _observer_forward_pre_hook(module, input):
    module.input_activation_post_process(input[0])
    return input


def _is_observer_machinery(child) -> bool:
    """Whether ``child`` is an embedded calibration module rather than a
    real submodule.  A wrapper carrying only such modules (for example a
    stub with its inner fake-quantize) still qualifies as a leaf."""
    return isinstance(child, (FakeQuantize, ObserverBase))


def _add_observer_(module, qconfig_spec=None):
    """Instantiate observers on each quantizable leaf module.

    ``activation_post_process`` records the module's output activations;
    ``input_activation_post_process`` records the incoming activations and
    ``weight_observer`` calibrates on the module's own weight.
    """
    # The dequantize boundary is swapped out at conversion and carries no
    # calibration of its own.
    if isinstance(module, DeQuantStub):
        return
    if getattr(module, "activation_post_process", None) is not None:
        # Already prepared; attaching again would duplicate the hooks.
        return
    is_leaf = all(_is_observer_machinery(child) for child in module.children())
    if is_leaf and getattr(module, "qconfig", None) is not None:
        if qconfig_spec is not None and not any(
            _matches(module, spec) for spec in qconfig_spec
        ):
            return
        module.activation_post_process = module.qconfig.activation()
        module.register_forward_hook(_observer_forward_hook, prepend=True)
        module.input_activation_post_process = module.qconfig.activation()
        module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)
        if getattr(module, "weight", None) is not None and module.qconfig.weight is not None:
            module.weight_observer = module.qconfig.weight()
            module.weight_observer(module.weight.detach())
        return
    for child in module.children():
        _add_observer_(child, qconfig_spec)


def prepare(model, inplace=False, qconfig_spec=None):
    """Attach observers to float modules for post-training calibration."""
    if not inplace:
        model = copy.deepcopy(model)
    propagate_qconfig_(model)
    _add_observer_(model, qconfig_spec)
    return model


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------



def _resolve_input_qparams(mod):
    observer = getattr(mod, "input_activation_post_process", None)
    if observer is None:
        return None
    return observer.calculate_qparams()


def _resolve_weight_qparams(mod):
    observer = getattr(mod, "weight_observer", None)
    if observer is None:
        return None
    return observer.calculate_qparams()


def convert(module, mapping=None, inplace=False, remove_qconfig=True):
    """Swap calibrated float modules for their quantized counterparts.

    Each target class must expose ``from_float``; activation qparams the
    quantized kernels require are resolved from the observers attached by
    :func:`prepare` and exposed on the float module.
    """
    if not inplace:
        module = copy.deepcopy(module)
    if mapping is None:
        mapping = get_default_static_quant_module_mappings()
    module = _convert(module, mapping)
    if remove_qconfig:
        _remove_qconfig(module)
    return module


def _convert(module, mapping):
    for name, child in module.named_children():
        new_child = _convert(child, mapping)
        if new_child is not child:
            module._modules[name] = new_child
    for source_type, target_type in mapping.items():
        if isinstance(module, source_type):
            return target_type.from_float(module)
    return module


def _remove_qconfig(module):
    if hasattr(module, "qconfig"):
        del module.qconfig
    # Modules that survive conversion run in the quantized regime from here
    # on; their float-era observer hooks would feed quantized tensors into
    # the calibration observers, so strip the whole attachment.
    for attr in ("activation_post_process", "input_activation_post_process",
                 "weight_observer"):
        if hasattr(module, attr):
            delattr(module, attr)
    for hook_id, hook in list(module._forward_pre_hooks.items()):
        if hook is _observer_forward_pre_hook:
            del module._forward_pre_hooks[hook_id]
    for hook_id, hook in list(module._forward_hooks.items()):
        if hook is _observer_forward_hook:
            del module._forward_hooks[hook_id]
    for child in module.children():
        _remove_qconfig(child)


# ---------------------------------------------------------------------------
# one-shot entry points
# ---------------------------------------------------------------------------


def quantize(model, run_fn, run_args, mapping=None, inplace=False):
    """Prepare, calibrate through ``run_fn(*run_args)`` and convert.

    Only submodules carrying a ``qconfig`` are quantized; attach qconfigs
    (or pass a ``qconfig_dict`` through :func:`propagate_qconfig_`) before
    calling.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepared = prepare(model, inplace=True)
    run_fn(prepared, *run_args)
    return convert(prepared, mapping=mapping, inplace=True)


def quantize_dynamic(model, qconfig_spec=None, mapping=None, inplace=False):
    """Swap modules for their dynamic-quantization counterparts.

    No calibration is performed: dynamic modules quantize activations at
    inference time and carry statically quantized weights.
    """
    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    _dynamic_swap(model, mapping, qconfig_spec)
    return model


def _dynamic_swap(module, mapping, qconfig_spec):
    for name, child in module.named_children():
        _dynamic_swap(child, mapping, qconfig_spec)
        swapped = _dynamic_target(child, mapping, qconfig_spec)
        if swapped is not child:
            module._modules[name] = swapped


def _dynamic_target(module, mapping, qconfig_spec):
    for source_type, target_type in mapping.items():
        if isinstance(module, source_type):
            if qconfig_spec is not None and not any(
                _matches(module, spec) for spec in qconfig_spec
            ):
                return module
            return target_type.from_float(module)
    return module


def add_quant_dequant(module):
    """Wrap a leaf float module with quant/dequant stubs.

    The stubs simulate the quantization round trip around the module, which
    surfaces quantization noise during float training.
    """
    qconfig = getattr(module, "qconfig", None)
    if qconfig is None:
        raise ValueError(
            "add_quant_dequant: the module must carry a qconfig attribute")
    quant_stub = QuantStub(qconfig.activation)
    dequant_stub = DeQuantStub()
    holder = nn.Sequential(quant_stub, module, dequant_stub)
    return holder
