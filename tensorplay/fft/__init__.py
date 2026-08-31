"""Fourier transforms and related frequency-domain utilities.

The package is organized so each family can grow into its own module:

* :mod:`~tensorplay.fft._transforms` -- the forward/inverse transform families
  (complex-to-complex, real-to-complex, complex-to-real, Hermitian).
* :mod:`~tensorplay.fft._frequency` -- frequency-grid construction and
  spectrum re-ordering.
* :mod:`~tensorplay.fft._helpers` -- axis/size/normalization bookkeeping
  shared by the transform families.
"""
from tensorplay import Tensor

from ._frequency import fftfreq, fftshift, ifftshift, rfftfreq
from ._transforms import (
    fft,
    fft2,
    fftn,
    hfft,
    hfft2,
    hfftn,
    ifft,
    ifft2,
    ifftn,
    ihfft,
    ihfft2,
    ihfftn,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftn,
)

__all__ = [
    "Tensor",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
