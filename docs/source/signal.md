# tensorplay.signal

Window functions for spectral analysis and filter design. A window is a
length-`M` sequence of weights that tapers a finite segment of a signal,
reducing the spectral leakage that comes from cutting the segment off sharply.

All window functions share a common shape of keyword arguments:

- ``M`` — the window length, a positive integer.
- ``sym`` — whether the window is symmetric (``True``, the default) or periodic
  (``False``). Symmetric windows are used in filter design; periodic windows are
  used for spectral analysis, where the window is applied to one period of a
  sampled signal and the first and last samples are not duplicates.
- ``dtype``, ``layout``, ``device``, ``requires_grad`` — the usual tensor
  construction arguments, so window tensors are created directly on the desired
  device with the desired data type.

```python
import tensorplay as tp
from tensorplay.signal import windows

print(windows.hann(8))
print(windows.blackman(8, sym=False))
# window tensors are ordinary tensors, so they can be moved or differentiated
w = windows.kaiser(64, beta=8.0)
```

## Window functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.signal.windows.bartlett
    tensorplay.signal.windows.blackman
    tensorplay.signal.windows.cosine
    tensorplay.signal.windows.exponential
    tensorplay.signal.windows.gaussian
    tensorplay.signal.windows.general_cosine
    tensorplay.signal.windows.general_hamming
    tensorplay.signal.windows.hamming
    tensorplay.signal.windows.hann
    tensorplay.signal.windows.kaiser
    tensorplay.signal.windows.nuttall
```

## Where to go next

- [FFT functions](fft.md) — windowing is typically a preamble to a spectral
  transform; the FFT page covers the transform side.
- [Tensors](tensorplay.md) — the tensor APIs the window functions are built on.