"""Spiking neural network primitives.

The module provides the neuron models, surrogate gradients, spike encoders,
and the functional primitives needed to build and train spiking networks.
The neuron dynamics run in fused native CPU/CUDA operators
(``snn_lif_step`` / ``snn_lif_step_backward``); this package wires them into
the autograd engine and the ``nn.Module`` API.

Example:
    >>> import tensorplay as tp
    >>> from tensorplay import snn
    >>> lif = snn.LIFNode(threshold=1.0, decay=0.5)
    >>> x = tp.rand(4, 16)
    >>> spike = lif(x)            # single-step: [4, 16] binary spikes
    >>> lif.reset()
    >>> seq = snn.LIFNode(step_mode="m")
    >>> out = seq(tp.rand(8, 4, 16))   # multi-step: [8, 4, 16] spike train
"""

from . import encoding as encoding
from . import functional as functional
from . import surrogate as surrogate
from .encoding import (
    poisson_encode,
    rate_encode,
    temporal_encode,
)
from .functional import heaviside, lif_seq, lif_step, spike, surrogate_grad
from .neuron import IFNode, LIFNode, ParametricLIFNode
from .surrogate import (
    FastSigmoidSurrogate,
    PiecewiseLinearSurrogate,
    SigmoidSurrogate,
    StraightThroughSurrogate,
)

__all__ = [
    # neurons
    "IFNode",
    "LIFNode",
    "ParametricLIFNode",
    # functional
    "heaviside",
    "spike",
    "surrogate_grad",
    "lif_step",
    "lif_seq",
    # surrogates
    "StraightThroughSurrogate",
    "SigmoidSurrogate",
    "FastSigmoidSurrogate",
    "PiecewiseLinearSurrogate",
    # encoders
    "rate_encode",
    "poisson_encode",
    "temporal_encode",
    # submodules
    "encoding",
    "functional",
    "surrogate",
]