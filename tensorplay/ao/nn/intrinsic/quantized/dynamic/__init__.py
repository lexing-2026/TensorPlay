"""Dynamically quantized fused operators."""

from ....quantized.dynamic import Linear


class LinearReLU(Linear):
    """Integer linear compute with a fused rectifier and floating output."""

    _FUSED_RELU = True


__all__ = ['LinearReLU']
