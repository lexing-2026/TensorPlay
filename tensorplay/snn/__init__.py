"""Spiking neural network primitives.

The module provides the neuron models, surrogate gradients, spike encoders,
monitoring tools, and the functional primitives needed to build and train
spiking networks.  The neuron dynamics run in fused native CPU/CUDA operators
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
from . import layer as layer
from . import monitor as monitor
from . import surrogate as surrogate
from .encoding import (
    GaussianTuningCurveEncoder,
    LatencyEncoder,
    PeriodicEncoder,
    WeightedPhaseEncoder,
    poisson_encode,
    rate_encode,
    temporal_encode,
)
from .functional import (
    heaviside,
    lif_seq,
    lif_step,
    spike,
    spike_rate,
    spike_rate_loss,
    surrogate_grad,
)
from .layer import (
    SeqToANNContainer,
    SpikingBatchNorm1d,
    SpikingBatchNorm2d,
    SpikingConv2d,
)
from .monitor import SpikeMonitor
from .neuron import ALIFNode, IFNode, LIFNode, ParametricLIFNode, SpikeLinear
from .surrogate import (
    ATanSurrogate,
    ErfSurrogate,
    FastSigmoidSurrogate,
    LeakyKRELUSurrogate,
    NonzeroSignLogAbsSurrogate,
    PiecewiseExpSurrogate,
    PiecewiseLeakyRELUSurrogate,
    PiecewiseLinearSurrogate,
    PiecewiseQuadraticSurrogate,
    QPseudoSpikeSurrogate,
    S2NNSurrogate,
    SigmoidSurrogate,
    SquarewaveFourierSeriesSurrogate,
    StraightThroughSurrogate,
    SuperSpikeSurrogate,
)

__all__ = [
    # neurons
    "IFNode",
    "LIFNode",
    "ParametricLIFNode",
    "ALIFNode",
    "SpikeLinear",
    # functional
    "heaviside",
    "spike",
    "surrogate_grad",
    "lif_step",
    "lif_seq",
    "spike_rate",
    "spike_rate_loss",
    # surrogates
    "StraightThroughSurrogate",
    "SigmoidSurrogate",
    "FastSigmoidSurrogate",
    "PiecewiseLinearSurrogate",
    "ATanSurrogate",
    "ErfSurrogate",
    "PiecewiseQuadraticSurrogate",
    "PiecewiseExpSurrogate",
    "SuperSpikeSurrogate",
    "NonzeroSignLogAbsSurrogate",
    "PiecewiseLeakyRELUSurrogate",
    "LeakyKRELUSurrogate",
    "S2NNSurrogate",
    "QPseudoSpikeSurrogate",
    "SquarewaveFourierSeriesSurrogate",
    # encoders
    "rate_encode",
    "poisson_encode",
    "temporal_encode",
    "PeriodicEncoder",
    "GaussianTuningCurveEncoder",
    "LatencyEncoder",
    "WeightedPhaseEncoder",
    # layers
    "SeqToANNContainer",
    "SpikingConv2d",
    "SpikingBatchNorm1d",
    "SpikingBatchNorm2d",
    # monitoring
    "SpikeMonitor",
    # submodules
    "encoding",
    "functional",
    "layer",
    "monitor",
    "surrogate",
]