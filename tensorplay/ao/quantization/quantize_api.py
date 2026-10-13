"""Config-driven model quantization entry point.

``quantize_(model, config)`` converts matching modules in-place according to
the config type. Config types register a ``(module, config) -> module``
transform handler with :func:`register_quantize_module_handler`, mirroring
the upstream config workflow while keeping every transform native.
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


def quantize_(
    model: "tp.nn.Module",
    config,
    filter_fn: Optional[Callable] = None,
    device=None,
) -> "tp.nn.Module":
    """Convert matching module weights in ``model`` in-place.

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
    for name, module in list(model.named_modules()):
        if filter_fn(module, name):
            handler(module, config)
    return model