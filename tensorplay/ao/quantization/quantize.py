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
import inspect

from tensorplay import nn

from .fake_quantize import FakeQuantize, FakeQuantizeBase
from .observer import ObserverBase, UniformQuantizationObserverBase
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

    def assign(current, inherited, prefix):
        config = qconfig_dict.get(type(current), qconfig_dict.get(type(current).__name__, inherited))
        config = qconfig_dict.get(prefix, config)
        current.qconfig = getattr(current, 'qconfig', config)
        for name, child in current.named_children():
            if not _is_observer_machinery(child):
                assign(child, current.qconfig, prefix + '.' + name if prefix else name)

    assign(module, qconfig_dict.get('', getattr(module, 'qconfig', None)), '')


# ---------------------------------------------------------------------------
# observer attachment
# ---------------------------------------------------------------------------


def _observer_forward_hook(module, input, output):
    module.activation_post_process.to(output.device)
    return module.activation_post_process(output)


def _observer_forward_pre_hook(module, input):
    module.input_activation_post_process.to(input[0].device)
    module.input_activation_post_process(input[0])
    return input


def _is_observer_machinery(child) -> bool:
    """Whether ``child`` is an embedded calibration module rather than a
    real submodule.  A wrapper carrying only such modules (for example a
    stub with its inner fake-quantize) still qualifies as a leaf."""
    return isinstance(child, (FakeQuantizeBase, ObserverBase))


def _add_observer_(module, qconfig_spec=None):
    """Instantiate observers on each quantizable leaf module.

    ``activation_post_process`` records the module's output activations;
    ``input_activation_post_process`` records the incoming activations and
    ``weight_observer`` calibrates on the module's own weight.
    """
    # The dequantize boundary is swapped out at conversion and carries no
    # calibration of its own.
    if isinstance(module, (DeQuantStub, nn.Identity)):
        return
    if getattr(module, "activation_post_process", None) is not None:
        # Already prepared; attaching again would duplicate the hooks.
        return
    from tensorplay.ao.nn.intrinsic import _FusedModule
    is_leaf = isinstance(module, _FusedModule) or all(_is_observer_machinery(child) for child in module.children())
    if is_leaf and getattr(module, "qconfig", None) is not None:
        if qconfig_spec is not None and not any(
            _matches(module, spec) for spec in qconfig_spec
        ):
            return
        module.activation_post_process = module.qconfig.activation()
        if getattr(module, '_qat_prepared', False) and isinstance(module.activation_post_process, UniformQuantizationObserverBase):
            module.activation_post_process = FakeQuantize(observer=module.activation_post_process)
        module.register_forward_hook(_observer_forward_hook, prepend=True)
        module.input_activation_post_process = module.qconfig.activation()
        module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)
        if getattr(module, "weight", None) is not None and module.qconfig.weight is not None and not hasattr(module, "weight_fake_quant"):
            module.weight_observer = module.qconfig.weight()
            module.weight_observer.to(module.weight.device)
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


def convert(module, mapping=None, inplace=False, remove_qconfig=True, use_precomputed_fake_quant=False):
    """Swap calibrated float modules for their quantized counterparts.

    Each target class must expose ``from_float``; activation qparams the
    quantized kernels require are resolved from the observers attached by
    :func:`prepare` and exposed on the float module.
    """
    if not inplace:
        module = copy.deepcopy(module)
    if mapping is None:
        mapping = get_default_static_quant_module_mappings()
    module = _convert(module, mapping, use_precomputed_fake_quant)
    if remove_qconfig:
        _remove_qconfig(module)
    return module


def _convert(module, mapping, use_precomputed_fake_quant=False):
    target_type = mapping.get(type(module))
    if target_type is not None and getattr(module, 'qconfig', None) is not None:
        kwargs = {}
        if 'use_precomputed_fake_quant' in inspect.signature(target_type.from_float).parameters:
            kwargs['use_precomputed_fake_quant'] = use_precomputed_fake_quant
        result = target_type.from_float(module, **kwargs)
        if hasattr(module, 'qconfig'):
            result.qconfig = module.qconfig
        for hook in module._forward_pre_hooks.values():
            if hook is not _observer_forward_pre_hook:
                result.register_forward_pre_hook(hook)
        for hook in module._forward_hooks.values():
            if hook is not _observer_forward_hook:
                result.register_forward_hook(hook)
        result.train(module.training)
        return result
    for name, child in list(module.named_children()):
        if not _is_observer_machinery(child):
            module._modules[name] = _convert(child, mapping, use_precomputed_fake_quant)
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


def quantize_dynamic(model, qconfig_spec=None, mapping=None, inplace=False, dtype=None):
    """Swap modules for their dynamic-quantization counterparts.

    No calibration is performed: dynamic modules quantize activations at
    inference time and carry statically quantized weights.
    """
    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    from tensorplay import qint8
    from .qconfig import default_dynamic_qconfig
    if dtype is not None and dtype != qint8:
        raise ValueError('Dynamic integer linear conversion requires signed 8-bit weights')
    if qconfig_spec is None:
        qconfig_spec = {source: default_dynamic_qconfig for source in mapping}
    elif not isinstance(qconfig_spec, dict):
        qconfig_spec = {source: default_dynamic_qconfig for source in qconfig_spec}
    model.eval()
    propagate_qconfig_(model, qconfig_spec)
    return convert(model, mapping=mapping, inplace=True)


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


def prepare_qat(model, mapping=None, inplace=False):
    from .quantization_mappings import get_default_qat_module_mappings
    if not model.training:
        raise ValueError('QAT preparation requires training mode')
    if not inplace:
        model = copy.deepcopy(model)
    propagate_qconfig_(model)
    model = convert(model, mapping=mapping or get_default_qat_module_mappings(),
                    inplace=True, remove_qconfig=False)
    for module in model.modules():
        if not _is_observer_machinery(module):
            module._qat_prepared = True
    return prepare(model, inplace=True)


def quantize_qat(model, run_fn, run_args, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    model = prepare_qat(model, inplace=True)
    run_fn(model, *run_args)
    model.eval()
    return convert(model, inplace=True)


__all__ += ['prepare_qat', 'quantize_qat']
