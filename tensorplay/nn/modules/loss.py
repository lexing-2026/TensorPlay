import tensorplay as tp
from .module import Module
from .. import functional as F
from typing import Optional


__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CTCLoss",
    "CrossEntropyLoss",
    "CosineEmbeddingLoss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]


class _Loss(Module):
    __constants__ = ['reduction']
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction


class _WeightedLoss(_Loss):
    __constants__ = ['reduction']
    weight: Optional[tp.Tensor]

    def __init__(self, weight: Optional[tp.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(reduction)
        self.weight = weight


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


class L1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__(reduction)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    __constants__ = ['reduction', 'beta']
    beta: float

    def __init__(self, reduction: str = 'mean', beta: float = 1.0) -> None:
        super(SmoothL1Loss, self).__init__(reduction)
        self.beta = beta

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(_Loss):
    __constants__ = ['reduction', 'delta']
    delta: float

    def __init__(self, reduction: str = 'mean', delta: float = 1.0) -> None:
        super(HuberLoss, self).__init__(reduction)
        self.delta = delta

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class KLDivLoss(_Loss):
    __constants__ = ['reduction', 'log_target']
    log_target: bool

    def __init__(self, reduction: str = 'mean', log_target: bool = False) -> None:
        super(KLDivLoss, self).__init__(reduction)
        self.log_target = log_target

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class BCELoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[tp.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    __constants__ = ['reduction']
    weight: Optional[tp.Tensor]
    pos_weight: Optional[tp.Tensor]

    def __init__(self, weight: Optional[tp.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[tp.Tensor] = None) -> None:
        super(BCEWithLogitsLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


class MarginRankingLoss(_Loss):
    __constants__ = ['margin']
    margin: float

    def __init__(self, margin: float = 0.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MarginRankingLoss, self).__init__(reduction)
        self.margin = margin

    def forward(self, input1: tp.Tensor, input2: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.reduction)


class HingeEmbeddingLoss(_Loss):
    __constants__ = ['margin']
    margin: float

    def __init__(self, margin: float = 1.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(HingeEmbeddingLoss, self).__init__(reduction)
        self.margin = margin

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.hinge_embedding_loss(input, target, self.margin, self.reduction)


class CosineEmbeddingLoss(_Loss):
    __constants__ = ['margin']
    margin: float

    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(CosineEmbeddingLoss, self).__init__(reduction)
        self.margin = margin

    def forward(self, input1: tp.Tensor, input2: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.reduction)


class SoftMarginLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(SoftMarginLoss, self).__init__(reduction)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.soft_margin_loss(input, target, reduction=self.reduction)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[tp.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MultiLabelSoftMarginLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.multilabel_soft_margin_loss(input, target, self.weight, reduction=self.reduction)


class MultiMarginLoss(_Loss):
    __constants__ = ['margin', 'p', 'reduction']
    margin: float
    p: int

    def __init__(self, p: int = 1, margin: float = 1.0, weight: Optional[tp.Tensor] = None,
                 size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MultiMarginLoss, self).__init__(reduction)
        if p not in (1, 2):
            raise ValueError("only p == 1 and p == 2 supported")
        self.p = p
        self.margin = margin
        self.weight = weight

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.multi_margin_loss(input, target, p=self.p, margin=self.margin,
                                   weight=self.weight, reduction=self.reduction)


class TripletMarginLoss(_Loss):
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(self, margin=1.0, p=2.0, eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction: str = 'mean'):
        super(TripletMarginLoss, self).__init__(reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor: tp.Tensor, positive: tp.Tensor, negative: tp.Tensor) -> tp.Tensor:
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)


class TripletMarginWithDistanceLoss(_Loss):
    __constants__ = ['margin', 'swap', 'reduction']
    margin: float
    swap: bool

    def __init__(self, *, distance_function=None, margin: float = 1.0, swap: bool = False,
                 reduction: str = 'mean'):
        super(TripletMarginWithDistanceLoss, self).__init__(reduction)
        self.distance_function = distance_function
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: tp.Tensor, positive: tp.Tensor, negative: tp.Tensor) -> tp.Tensor:
        return F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                                   distance_function=self.distance_function,
                                                   margin=self.margin, swap=self.swap,
                                                   reduction=self.reduction)


class PoissonNLLLoss(_Loss):
    __constants__ = ['log_input', 'full', 'eps', 'reduction']
    log_input: bool
    full: bool
    eps: float

    def __init__(self, *, log_input: bool = True, full: bool = False, eps: float = 1e-8,
                 reduction: str = 'mean') -> None:
        super(PoissonNLLLoss, self).__init__(reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, log_input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.poisson_nll_loss(log_input, target, log_input=self.log_input, full=self.full,
                                  eps=self.eps, reduction=self.reduction)


class GaussianNLLLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: tp.Tensor, target: tp.Tensor, var: tp.Tensor) -> tp.Tensor:
        return F.gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps,
                                   reduction=self.reduction)


class MultiLabelMarginLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MultiLabelMarginLoss, self).__init__(reduction)

    def forward(self, input: tp.Tensor, target: tp.Tensor) -> tp.Tensor:
        return F.multilabel_margin_loss(input, target, reduction=self.reduction)


class CTCLoss(_Loss):
    __constants__ = ['blank', 'reduction']
    blank: int
    zero_infinity: bool

    def __init__(self, blank: int = 0, reduction: str = 'mean', zero_infinity: bool = False) -> None:
        super(CTCLoss, self).__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(self,
                log_probs: tp.Tensor,
                targets: tp.Tensor,
                input_lengths: tp.Tensor,
                target_lengths: tp.Tensor) -> tp.Tensor:
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          self.blank, self.reduction, self.zero_infinity)
