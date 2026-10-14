"""Latency, size and accuracy benchmarking for quantized models.

Small helpers to quantify what quantization buys on a given model: model
parameter size, average inference latency, and per-output similarity between a
float model and its quantized counterpart.  The comparisons are deliberately
simple so they run on CPU without a dataloader.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import tensorplay as tp
from tensorplay import nn

__all__ = [
    "model_size_bytes",
    "benchmark_latency",
    "compare_outputs",
    "benchmark_quantized",
    "quantize_input_for_model",
]


def model_size_bytes(model: nn.Module) -> int:
    """Return the total parameter storage size of ``model`` in bytes."""
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()
    return total


def benchmark_latency(
    model: nn.Module,
    example_input,
    num_iterations: int = 100,
    warmup: int = 10,
    repeat: int = 3,
) -> float:
    """Return the average inference latency in milliseconds.

    ``example_input`` can be a single tensor or a tuple of positional args.
    The model is run ``repeat`` times over ``num_iterations`` forward passes
    each; the minimum mean latency across repeats is returned.
    """
    if not isinstance(example_input, tuple):
        example_input = (example_input,)
    best = float("inf")
    with tp.no_grad():
        for _ in range(warmup):
            model(*example_input)
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(num_iterations):
                model(*example_input)
            elapsed = (time.perf_counter() - start) / num_iterations
            best = min(best, elapsed)
    return best * 1e3


def _has_input_quantizer(model: nn.Module) -> bool:
    """Return whether the model contains a QuantStub/Quantize module."""
    from tensorplay.ao.nn.quantized import Quantize
    from tensorplay.ao.quantization.stubs import QuantStub

    for module in model.modules():
        if isinstance(module, (QuantStub, Quantize)):
            return True
    return False


def _is_quantized_model(model: nn.Module) -> bool:
    """Return whether the model contains quantized parameters or packed data."""
    for module in model.modules():
        if hasattr(module, "_packed_params"):
            return True
        for param in module.parameters(recurse=False):
            if tp.is_quantized(param):
                return True
    return False


def quantize_input_for_model(model: nn.Module, example_input, dtype=tp.qint8):
    """Quantize ``example_input`` with a symmetric scale derived from its range.

    Used when a quantized model has no input ``QuantStub`` and therefore
    expects a quantized tensor.  ``dtype`` defaults to ``qint8``.
    """
    if isinstance(example_input, tuple):
        return tuple(
            quantize_input_for_model(model, x, dtype) if isinstance(x, tp.Tensor) else x
            for x in example_input
        )
    if not isinstance(example_input, tp.Tensor):
        return example_input
    if example_input.dtype in (tp.qint8, tp.quint8, tp.qint32):
        return example_input
    scale = (example_input.abs().max().item() / 127.0) or 1.0
    return tp.quantize_per_tensor(example_input, scale, 0, dtype)


def _model_inputs(model: nn.Module, example_input):
    """Prepare float or quantized inputs depending on the model's I/O style."""
    if isinstance(example_input, tuple):
        return example_input
    if _has_input_quantizer(model) or not _is_quantized_model(model):
        return (example_input,)
    return (quantize_input_for_model(model, example_input),)


def compare_outputs(
    model_a: nn.Module,
    model_b: nn.Module,
    example_input,
    metric_fn: Callable = lambda a, b: (a - b).abs().mean().item(),
) -> float:
    """Run both models on the same input and return the metric value."""
    inputs_a = _model_inputs(model_a, example_input)
    inputs_b = _model_inputs(model_b, example_input)
    with tp.no_grad():
        out_a = model_a(*inputs_a)
        out_b = model_b(*inputs_b)
    if isinstance(out_a, tp.Tensor) and tp.is_quantized(out_a):
        out_a = tp.dequantize(out_a)
    if isinstance(out_b, tp.Tensor) and tp.is_quantized(out_b):
        out_b = tp.dequantize(out_b)
    return metric_fn(out_a, out_b)


def benchmark_quantized(
    float_model: nn.Module,
    quantized_model: nn.Module,
    example_input,
    num_iterations: int = 100,
) -> dict[str, Any]:
    """Compare a float model against its quantized counterpart.

    Returns a dictionary with ``float_latency_ms``, ``quantized_latency_ms``,
    ``speedup``, ``float_size_bytes``, ``quantized_size_bytes``, ``size_ratio``
    and ``output_metric`` (default: mean absolute error).
    """
    float_inputs = _model_inputs(float_model, example_input)
    quantized_inputs = _model_inputs(quantized_model, example_input)
    float_latency = benchmark_latency(float_model, float_inputs, num_iterations)
    quantized_latency = benchmark_latency(quantized_model, quantized_inputs, num_iterations)
    float_size = model_size_bytes(float_model)
    quantized_size = model_size_bytes(quantized_model)
    metric = compare_outputs(float_model, quantized_model, example_input)
    return {
        "float_latency_ms": float_latency,
        "quantized_latency_ms": quantized_latency,
        "speedup": float_latency / quantized_latency if quantized_latency > 0 else float("inf"),
        "float_size_bytes": float_size,
        "quantized_size_bytes": quantized_size,
        "size_ratio": quantized_size / float_size if float_size > 0 else 1.0,
        "output_metric": metric,
    }