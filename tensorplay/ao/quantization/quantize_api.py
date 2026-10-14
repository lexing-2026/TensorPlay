"""Config-driven model quantization entry point.

``quantize_(model, config)`` converts matching modules in-place according to
the config type. Config types register a ``(module, config) -> module``
transform handler with :func:`register_quantize_module_handler`, keeping
every transform built on native modules and parameters.
"""

from typing import Callable, Optional

import tensorplay as tp


_QUANTIZE_MODULE_HANDLERS: dict = {}


def register_quantize_module_handler(config_type):
    """Register a ``(module, config) -> module`` transform for ``config_type``."""

    def decorator(fn):
        _QUANTIZE_MODULE_HANDLERS[config_type] = fn
        return fn

    return decorator


def _is_linear(module, name) -> bool:
    del name
    return isinstance(module, tp.nn.Linear)


def _set_module_at(model, fqn, replacement):
    """Assign ``replacement`` to the module at dotted ``fqn`` in ``model``."""
    if "." in fqn:
        parent_name, child_name = fqn.rsplit(".", 1)
        parent = model
        for part in parent_name.split("."):
            parent = getattr(parent, part)
        setattr(parent, child_name, replacement)
    else:
        setattr(model, fqn, replacement)


def quantize_(
    model: "tp.nn.Module",
    config,
    filter_fn: Optional[Callable] = None,
    device=None,
) -> "tp.nn.Module":
    """Convert matching module weights in ``model`` in-place.

    Handlers registered through :func:`register_quantize_module_handler` may
    mutate the module in place (returning the same object) or return a
    replacement module, which is then re-attached at the module's location.

    Args:
        model: the module tree to convert.
        config: a registered quantization config, e.g. ``bitnet_training()``
            or ``int8_mixed_precision_training()``.
        filter_fn: optional ``(module, fqn) -> bool`` predicate; defaults to
            matching ``nn.Linear`` modules.
        device: optional device to move the model to before conversion.
    """
    handler = _QUANTIZE_MODULE_HANDLERS.get(type(config))
    if handler is None:
        raise TypeError(
            "quantize_ has no registered handler for config "
            f"{type(config).__name__}; use a config created by e.g. "
            "bitnet_training() or int8_mixed_precision_training()"
        )
    filter_fn = _is_linear if filter_fn is None else filter_fn
    if device is not None:
        model = model.to(device)
    result = model
    for name, module in list(model.named_modules()):
        if filter_fn(module, name):
            replacement = handler(module, config)
            if replacement is not None and replacement is not module:
                if name == "":
                    # The whole model is being quantized; return the
                    # replacement module itself.
                    result = replacement
                else:
                    _set_module_at(model, name, replacement)
    return result