"""Linear training for a dynamic activation grid."""

from ..modules.linear import Linear as StaticLinear
from ....quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class Linear(StaticLinear):
    def __init__(self, in_features, out_features, bias=True, qconfig=None,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        activation = self.qconfig.activation()
        observer = getattr(activation, 'activation_post_process', activation)
        if not isinstance(observer, (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)) or observer.averaging_constant != 1:
            raise ValueError('Dynamic QAT requires a moving average observer with averaging constant equal to one')


__all__ = ['Linear']
