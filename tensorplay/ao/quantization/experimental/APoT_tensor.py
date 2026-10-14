"""APoT quantized tensor container."""

from .quantizer import APoTQuantizer


class TensorAPoT:
    """A tensor stored in APoT integer representation."""

    def __init__(self, quantizer: APoTQuantizer, apot_data):
        self.quantizer = quantizer
        self.data = apot_data

    def int_repr(self):
        return self.data