"""Operator wrappers carrying fixed output qparams.

``QFunctional`` instances hold scale/zero_point attributes (populated by the
conversion step) so quantized elementwise ops can be expressed as module
calls.  ``FloatFunctional`` performs the float-domain equivalents for use in
quantization-aware training.
"""

from __future__ import annotations

from tensorplay import nn

from . import functional as F

__all__ = ["FloatFunctional", "QFunctional", "FXFloatFunctional"]


class FloatFunctional(nn.Module):
    """Float-domain operation wrappers with the quantized call signature."""

    def __init__(self):
        super().__init__()

    def add(self, x, y):
        return x + y

    def mul(self, x, y):
        return x * y

    def cat(self, tensors, dim=0):
        import tensorplay

        return tensorplay.cat(tensors, dim)

    def add_relu(self, x, y):
        return (x + y).relu()

    def add_scalar(self, x, scalar):
        return x + scalar

    def mul_scalar(self, x, scalar):
        return x * scalar

    def matmul(self, x, y):
        import tensorplay

        return tensorplay.matmul(x, y)

    def forward(self, x, y):
        return self.add(x, y)


class QFunctional(nn.Module):
    """Wrapper class for quantized operations with fixed output qparams.

    The instance can be used instead of the functional namespace, reading the
    declared ``scale`` and ``zero_point``:

        >>> q_add = QFunctional()  # doctest: +SKIP
        >>> q_add.scale, q_add.zero_point = 1.0, 0  # doctest: +SKIP
        >>> q_add.add(a, b)  # doctest: +SKIP

    Valid operation names: ``add``, ``sub``, ``mul``, ``div``, ``cat``,
    ``add_relu``, ``add_scalar``, ``mul_scalar``.
    """

    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        from tensorplay.ao.quantization.stubs import QuantStub, DeQuantStub  # noqa: F401

        scale, zero_point = mod.activation_post_process.calculate_qparams()
        result = cls()
        result.scale = float(scale)
        result.zero_point = int(zero_point)
        return result

    def _fixed(self):
        return float(self.scale), int(self.zero_point)

    def add(self, a, b):
        scale, zero_point = self._fixed()
        return F.add(a, b, scale, zero_point)

    def sub(self, a, b):
        scale, zero_point = self._fixed()
        return F.sub(a, b, scale, zero_point)

    def mul(self, a, b):
        scale, zero_point = self._fixed()
        return F.mul(a, b, scale, zero_point)

    def div(self, a, b):
        scale, zero_point = self._fixed()
        return F.div(a, b, scale, zero_point)

    def cat(self, tensors, dim=0):
        scale, zero_point = self._fixed()
        return F.cat(tensors, dim, scale, zero_point)

    def add_relu(self, a, b):
        scale, zero_point = self._fixed()
        return F.add_relu(a, b, scale, zero_point)

    def add_scalar(self, x, scalar):
        return F.add_scalar(x, scalar)

    def mul_scalar(self, x, scalar):
        return F.mul_scalar(x, scalar)

    def matmul(self, x, y):
        return F.matmul(x, y, *self._fixed())

    def forward(self, x, y):
        return self.add(x, y)


class FXFloatFunctional(FloatFunctional):
    """Floating operation wrappers for graph-captured workflows."""
