"""Bias correction for post-training quantization."""

import tensorplay as tp
import tensorplay.ao.nn.quantized as nnq
import tensorplay.ao.ns._numeric_suite as ns
import tensorplay.ao.quantization
from tensorplay import nn

__all__ = [
    "get_module",
    "parent_child_names",
    "get_param",
    "MeanShadowLogger",
    "bias_correction",
]

_supported_modules = {nn.Linear, nn.Conv2d}
_supported_modules_quantized = {nnq.Linear, nnq.Conv2d}


def get_module(model, name):
    """Return the submodule with the given fully qualified name."""
    return dict(model.named_modules())[name]


def parent_child_names(name):
    """Split a submodule fqn into parent fqn and child name."""
    split_name = name.rsplit(".", 1)
    if len(split_name) == 1:
        return "", split_name[0]
    return split_name[0], split_name[1]


def get_param(module, attr):
    """Read a parameter, calling it if it is exposed as a method."""
    param = getattr(module, attr, None)
    if callable(param):
        return param()
    return param


class MeanShadowLogger(ns.ShadowLogger):
    """Logger that records the mean of the observed tensor."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.sum = 0

    def forward(self, x):
        x = x.detach()
        self.count += 1
        self.sum += x.mean().item()
        return x

    def get_mean(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def _get_float_model_output(module, input):
    """Run a float module and record the mean output via a shadow logger."""
    logger = MeanShadowLogger()
    module.register_forward_hook(logger)
    with tp.no_grad():
        module(input)
    module._forward_hooks.popitem()
    return logger.get_mean()


def bias_correction(float_model, quantized_model, data_loader, num_batches=2):
    """Correct the bias of quantized linear/conv layers using float statistics."""
    if not hasattr(float_model, "named_modules"):
        raise TypeError("float_model must be an nn.Module")
    if not hasattr(quantized_model, "named_modules"):
        raise TypeError("quantized_model must be an nn.Module")

    for name, module in quantized_model.named_modules():
        if type(module) not in _supported_modules_quantized:
            continue
        float_module = get_module(float_model, name)
        if type(float_module) not in _supported_modules:
            continue
        bias = get_param(module, "bias")
        if bias is None:
            continue
        error_sum = tp.zeros_like(bias)
        count = 0
        for batch_idx, (input, _target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            float_mean = _get_float_model_output(float_module, input)
            q_out = module(input)
            if q_out.is_quantized():
                q_out = q_out.dequantize()
            quantized_mean = q_out.mean(dim=(0, *range(2, q_out.dim())))
            error_sum += float_mean - float(quantized_mean.mean())
            count += 1
        if count:
            correction = error_sum / count
            module.set_weight_bias(get_param(module, "weight"), bias + correction)