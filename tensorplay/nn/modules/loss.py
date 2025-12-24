import tensorplay as tp
from .module import Module
from .. import functional as F
from typing import Optional


__all__ = ["MSELoss", "CrossEntropyLoss", "NLLLoss"]


class _Loss(Module):
    __constants__ = ['reduction']
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction

class MSELoss(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(reduction)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)

class CrossEntropyLoss(_Loss):
    __constants__ = ['ignore_index', 'label_smoothing']
    ignore_index: int
    label_smoothing: float
    weight: Optional[tp.Tensor]

    def __init__(self, weight: Optional[tp.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(CrossEntropyLoss, self).__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

class NLLLoss(_Loss):
    __constants__ = ['ignore_index']
    ignore_index: int
    weight: Optional[tp.Tensor]

    def __init__(self, weight: Optional[tp.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(NLLLoss, self).__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.nll_loss(input, target, weight=self.weight,
                          ignore_index=self.ignore_index, reduction=self.reduction)
