# Ported from torch/nn/modules/adaptive.py, adapted to tensorplay primitives
# (no gather/index_copy_/index_fill_: per-row column picks go through
# embedding-on-flattened, cluster contributions are masked instead of scattered).

import itertools
from collections import namedtuple
from collections.abc import Sequence

import tensorplay as tp
from tensorplay._C import DType
import tensorplay.nn.functional as F
from ... import Tensor
from .container import ModuleList, Sequential
from .linear import Linear
from .module import Module


__all__ = ["AdaptiveLogSoftmaxWithLoss"]

_ASMoutput = namedtuple("_ASMoutput", ["output", "loss"])


def _pick_columns(logprob: Tensor, index: Tensor) -> Tensor:
    # logprob: (N, C); index: (N,) int64 -> (N,), one entry per row.
    # tp.embedding(weight, indices) on the flattened log-probs == gather.
    n, c = logprob.shape[0], logprob.shape[1]
    flat = logprob.contiguous().reshape(-1)
    gid = tp.arange(n, dtype=DType.int64, device=logprob.device) * c + index.to(DType.int64)
    return tp.embedding(flat, gid)


class AdaptiveLogSoftmaxWithLoss(Module):
    (
        """Efficient softmax approximation.

    As described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Ciss\u00e9, David Grangier, and Herv\u00e9 J\u00e9gou
    <https://arxiv.org/abs/1609.04309>`__.
"""
        r"""
    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted according to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \texttt{in\_features})` or :math:`(\texttt{in\_features})`
        - target: :math:`(N)` or :math:`()` where each value satisfies :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)` or :math:`()`
        - output2: ``Scalar``

    .. _Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """
    )

    in_features: int
    n_classes: int
    cutoffs: list[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.0,
        head_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        cutoffs = list(cutoffs)

        if len(cutoffs) == 0:
            raise ValueError("cutoffs should be a sequence of length larger than 0")

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) > (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(
            self.in_features, self.head_size, bias=self.head_bias, **factory_kwargs
        )
        self.tail = ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False, **factory_kwargs),
                Linear(hsz, osz, bias=False, **factory_kwargs),
            )

            self.tail.append(projection)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput:
        """
        Runs the forward pass.
        """
        targ_dim = target_.dim()

        if targ_dim == 1:
            if input_.size(0) != target_.size(0):
                raise RuntimeError(
                    "Input and target should have the same size in the batch dimension."
                )
            if input_.dim() != 2:
                raise RuntimeError(
                    f"1D target tensor expects 2D input tensors, "
                    f"but found inputs with size {input_.size()}"
                )
        elif targ_dim == 0:
            if input_.dim() != 1:
                raise RuntimeError(
                    f"0D target tensor expects 1D input tensors, "
                    f"but found inputs with size {input_.size()}"
                )
        else:
            raise RuntimeError(
                "0D or 1D target tensor expected, multi-target not supported"
            )

        is_batched = targ_dim > 0
        input = input_ if is_batched else input_.unsqueeze(0)
        target = target_ if is_batched else target_.unsqueeze(0)
        if target.dtype != DType.int64:
            target = target.to(DType.int64)

        batch_size = target.size(0)
        device = input.device

        output = tp.zeros([batch_size], dtype=input.dtype, device=device)

        # Head rows pick the target column itself; cluster rows pick the
        # cluster's head column (shortlist_size + i - 1).
        gather_inds = target
        in_shortlist = target < self.shortlist_size

        used_rows = int(in_shortlist.to(DType.int64).sum().item())

        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)

        # Shortlist contribution: head_logprob[row, target[row]] on head rows.
        shortlist_pick = _pick_columns(head_logprob, target.clamp(max=self.shortlist_size - 1))
        output = output + tp.where(in_shortlist, shortlist_pick, 0.0)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 2):
            low_idx = cutoff_values[i + 1]
            high_idx = cutoff_values[i + 2]

            target_mask = (target >= low_idx) & (target < high_idx)
            used_rows += int(target_mask.to(DType.int64).sum().item())

            cluster_index = self.shortlist_size + i
            gather_inds = tp.where(
                target_mask, tp.full([batch_size], cluster_index, dtype=DType.int64, device=device),
                gather_inds)

            # Full-batch evaluation of the tail projection; masked rows carry
            # the cluster's local log-probability, others contribute zero.
            relative_target = (target - low_idx).clamp(min=0, max=high_idx - low_idx - 1)
            cluster_output = self.tail[i](input)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            local_logprob = _pick_columns(cluster_logprob, relative_target)
            output = output + tp.where(target_mask, local_logprob, 0.0)

        if used_rows != batch_size:
            raise RuntimeError(
                f"Target values should be in [0, {self.n_classes - 1}], "
                "but some values were out of range. "
            )

        output = output + _pick_columns(head_logprob, gather_inds)
        loss = (-output).mean()

        if not is_batched:
            output = output.squeeze(0)

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """Given input tensor, and output of ``self.head``, compute the log of the full distribution."""
        head_logprob = F.log_softmax(head_output, dim=1)
        out = tp.zeros(
            [head_output.size(0), self.n_classes], dtype=head_logprob.dtype, device=head_logprob.device
        )
        out[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(itertools.pairwise(self.cutoffs)):
            cluster_output = self.tail[i](input)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[
                :, self.shortlist_size + i
            ].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        r"""Compute log probabilities for all :math:`\texttt{n\_classes}`.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """
        head_output = self.head(input)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        r"""Return the class with the highest probability for each example in the input minibatch.

        This is equivalent to ``self.log_prob(input).argmax(dim=1)``, but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """
        head_output = self.head(input)
        output = tp.argmax(head_output, dim=1)
        not_in_shortlist = output >= self.shortlist_size
        if not bool(not_in_shortlist.any()):
            return output
        log_prob = self._get_full_log_prob(input, head_output)
        return tp.argmax(log_prob, dim=1)
