"""Concrete pruning methods and their functional entry points.

Each method class implements one recipe for zeroing entries or whole
channels of a tensor. The module-level functions apply a recipe to a named
parameter of a module in place: they install the mask reparameterization
described in :mod:`tensorplay.ao.pruning.base_classes` and return the
modified module.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import tensorplay
from tensorplay import Tensor, nn

from .base_classes import BasePruningMethod, PruningContainer
from .utils import (
    _compute_norm,
    _validate_pruning_amount_init,
    _validate_pruning_dim,
    _validate_structured_pruning,
    compute_nparams_to_prune,
    validate_pruning_amount,
)

__all__ = [
    "Identity",
    "RandomUnstructured",
    "L1Unstructured",
    "RandomStructured",
    "LnStructured",
    "CustomFromMask",
    "identity",
    "random_unstructured",
    "l1_unstructured",
    "random_structured",
    "ln_structured",
    "global_unstructured",
    "custom_from_mask",
    "remove",
    "is_pruned",
]


class Identity(BasePruningMethod):
    """Prune nothing and only install the mask reparameterization.

    The generated mask is a tensor of ones, which is useful to prepare a
    module for later iterative pruning without removing any unit yet.
    """

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        return default_mask

    @classmethod
    def apply(cls, module: nn.Module, name: str) -> BasePruningMethod:
        """Install the identity (all-ones mask) reparameterization.

        Args:
            module: module containing the tensor to reparameterize.
            name: parameter name within ``module`` on which pruning acts.
        """
        return super().apply(module, name)


class RandomUnstructured(BasePruningMethod):
    """Zero out a uniformly random subset of the currently unpruned units.

    Args:
        amount: quantity of units to prune. A float in ``[0, 1]`` denotes
            the fraction of units to prune; an int denotes the absolute
            number of units to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount: int | float) -> None:
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.numel()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = compute_nparams_to_prune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=tensorplay.contiguous_format)

        if nparams_toprune != 0:  # nothing to do when k=0
            # draw one uniform sample per unit; the positions of the k largest
            # samples form the random subset selected for pruning
            prob = tensorplay.rand_like(t)
            topk_indices = tensorplay.topk(prob.view(-1), k=nparams_toprune)[1]
            mask.view(-1)[topk_indices] = 0

        return mask

    @classmethod
    def apply(cls, module: nn.Module, name: str, amount: int | float) -> BasePruningMethod:
        """Install random unstructured pruning for ``module[name]``.

        Args:
            module: module containing the tensor to prune.
            name: parameter name within ``module`` on which pruning acts.
            amount: quantity of units to prune. A float in ``[0, 1]`` denotes
                the fraction of units to prune; an int denotes the absolute
                number of units to prune.
        """
        return super().apply(module, name, amount=amount)


class L1Unstructured(BasePruningMethod):
    """Zero out the units with the smallest magnitudes.

    Ranks all currently unpruned units by absolute value and removes the
    ``amount`` smallest ones.

    Args:
        amount: quantity of units to prune. A float in ``[0, 1]`` denotes
            the fraction of units to prune; an int denotes the absolute
            number of units to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount: int | float) -> None:
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.numel()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = compute_nparams_to_prune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=tensorplay.contiguous_format)

        if nparams_toprune != 0:  # nothing to do when k=0
            # select the k units with the smallest absolute values
            topk_indices = tensorplay.topk(
                tensorplay.abs(t).view(-1), k=nparams_toprune, largest=False
            )[1]
            # topk yields both values and indices; only the indices matter here
            mask.view(-1)[topk_indices] = 0

        return mask

    @classmethod
    def apply(
        cls,
        module: nn.Module,
        name: str,
        amount: int | float,
        importance_scores: Tensor | None = None,
    ) -> BasePruningMethod:
        """Install magnitude-based unstructured pruning for ``module[name]``.

        Args:
            module: module containing the tensor to prune.
            name: parameter name within ``module`` on which pruning acts.
            amount: quantity of units to prune. A float in ``[0, 1]`` denotes
                the fraction of units to prune; an int denotes the absolute
                number of units to prune.
            importance_scores: tensor of importance scores with the same
                shape as the parameter; each entry ranks the corresponding
                element of the parameter. When unspecified, the parameter
                itself is used.
        """
        return super().apply(
            module, name, amount=amount, importance_scores=importance_scores
        )


class RandomStructured(BasePruningMethod):
    """Zero out entire randomly selected channels of a tensor.

    Args:
        amount: quantity of channels to prune. A float in ``[0, 1]`` denotes
            the fraction of channels to prune; an int denotes the absolute
            number of channels to prune.
        dim: axis along which channels are defined. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount: int | float, dim: int = -1) -> None:
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """Compute a channel mask for ``t`` by drawing channels at random.

        Args:
            t: tensor whose channels are pruned.
            default_mask: mask accumulated from previous pruning iterations;
                must be respected by the new mask. Same shape as ``t``.

        Returns:
            The mask to apply to ``t``, with the same shape as ``t``.

        Raises:
            IndexError: if ``self.dim`` is not a valid axis of ``t``.
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)

        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = compute_nparams_to_prune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        validate_pruning_amount(nparams_toprune, tensor_size)

        # Compute the channel-level mask from k uniformly drawn ranks, then
        # broadcast it over every other axis of the tensor.
        def make_mask(t: Tensor, dim: int, nchannels: int, nchannels_toprune: int) -> Tensor:
            # generate a random number in [0, 1] to associate to each channel
            prob = tensorplay.rand(nchannels)
            # zero the channels holding the k = nchannels_toprune lowest draws
            threshold, _ = tensorplay.kthvalue(prob, k=nchannels_toprune)
            channel_mask = prob > threshold

            mask = tensorplay.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = channel_mask
            slc = tuple(slc)
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # nothing to do when k=0
            mask = default_mask
        else:
            # apply the new structured mask on top of prior (potentially
            # unstructured) mask
            mask = make_mask(t, self.dim, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(
        cls, module: nn.Module, name: str, amount: int | float, dim: int = -1
    ) -> BasePruningMethod:
        """Install random structured pruning for ``module[name]``.

        Args:
            module: module containing the tensor to prune.
            name: parameter name within ``module`` on which pruning acts.
            amount: quantity of channels to prune. A float in ``[0, 1]``
                denotes the fraction of channels to prune; an int denotes the
                absolute number of channels to prune.
            dim: axis along which channels are defined. Default: -1.
        """
        return super().apply(module, name, amount=amount, dim=dim)


class LnStructured(BasePruningMethod):
    """Zero out the channels with the smallest L``n``-norm.

    Reduces the tensor to one L``n``-norm per channel along ``dim`` and
    keeps the ``amount``-largest channels, zeroing the rest.

    Args:
        amount: quantity of channels to prune. A float in ``[0, 1]`` denotes
            the fraction of channels to prune; an int denotes the absolute
            number of channels to prune.
        n: norm order; accepts the orders valid for
            :func:`tensorplay.linalg.vector_norm` and
            :func:`tensorplay.linalg.matrix_norm` (e.g. a positive number,
            ``float('inf')``, ``float('-inf')``, ``'fro'``, ``'nuc'``).
        dim: axis along which channels are defined. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount: int | float, n: int | float | str, dim: int = -1) -> None:
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """Compute a channel mask for ``t`` by per-channel L``n``-norm.

        Args:
            t: tensor whose channels are pruned.
            default_mask: mask accumulated from previous pruning iterations;
                must be respected by the new mask. Same shape as ``t``.

        Returns:
            The mask to apply to ``t``, with the same shape as ``t``.

        Raises:
            IndexError: if ``self.dim`` is not a valid axis of ``t``.
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = compute_nparams_to_prune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        norm = _compute_norm(t, self.n, self.dim)
        # keep the k channels with the largest norms along dim=self.dim
        topk_indices = tensorplay.topk(norm, k=nparams_tokeep, largest=True)[1]
        # topk yields both values and indices; only the indices matter here

        # Compute the binary mask by starting from all 0s and filling in
        # 1s wherever topk_indices indicates, along self.dim. The mask has
        # the same shape as the tensor t.
        def make_mask(t: Tensor, dim: int, indices: Tensor) -> Tensor:
            # init mask to 0
            mask = tensorplay.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            slc = tuple(slc)
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # nothing to do when k=0
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk_indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask

    @classmethod
    def apply(
        cls,
        module: nn.Module,
        name: str,
        amount: int | float,
        n: int | float | str,
        dim: int,
        importance_scores: Tensor | None = None,
    ) -> BasePruningMethod:
        """Install norm-based structured pruning for ``module[name]``.

        Args:
            module: module containing the tensor to prune.
            name: parameter name within ``module`` on which pruning acts.
            amount: quantity of channels to prune. A float in ``[0, 1]``
                denotes the fraction of channels to prune; an int denotes the
                absolute number of channels to prune.
            n: norm order; accepts the orders valid for
                :func:`tensorplay.linalg.vector_norm` and
                :func:`tensorplay.linalg.matrix_norm`.
            dim: axis along which channels are defined.
            importance_scores: tensor of importance scores with the same
                shape as the parameter; each entry ranks the corresponding
                element of the parameter. When unspecified, the parameter
                itself is used.
        """
        return super().apply(
            module,
            name,
            amount=amount,
            n=n,
            dim=dim,
            importance_scores=importance_scores,
        )


class CustomFromMask(BasePruningMethod):
    """Zero out exactly the units designated by a caller-supplied mask.

    Args:
        mask: binary mask whose zeros mark the units to prune.
    """

    PRUNING_TYPE = "global"

    def __init__(self, mask: Tensor) -> None:
        self.mask = mask

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        if default_mask.shape != self.mask.shape:
            raise AssertionError(
                f"default_mask shape {default_mask.shape} must match "
                f"self.mask shape {self.mask.shape}"
            )
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module: nn.Module, name: str, mask: Tensor) -> BasePruningMethod:
        """Install a user-provided mask for ``module[name]``.

        Args:
            module: module containing the tensor to prune.
            name: parameter name within ``module`` on which pruning acts.
            mask: binary mask to be applied to the parameter.
        """
        return super().apply(module, name, mask=mask)


def identity(module: nn.Module, name: str) -> nn.Module:
    """Attach the pruning reparameterization to ``module[name]`` without
    pruning any unit.

    Modifies the module in place (and also returns it) by:

    1) adding a named buffer called ``name + '_mask'`` holding the binary
       mask applied to the parameter ``name``;
    2) replacing the parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Note:
        The mask is a tensor of ones.

    Args:
        module: module containing the tensor to prune.
        name: parameter name within ``module`` on which pruning acts.

    Returns:
        The modified (i.e. pruned) module.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = identity(nn.Linear(2, 3), "bias")
        >>> print(m.bias_mask)
        tensor([1., 1., 1.])
    """
    Identity.apply(module, name)
    return module


def random_unstructured(module: nn.Module, name: str, amount: int | float) -> nn.Module:
    """Prune ``module[name]`` by removing a random subset of its units.

    Removes ``amount`` (currently unpruned) units chosen uniformly at random.
    Modifies the module in place (and also returns it) by:

    1) adding a named buffer called ``name + '_mask'`` holding the binary
       mask applied to the parameter ``name``;
    2) replacing the parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Args:
        module: module containing the tensor to prune.
        name: parameter name within ``module`` on which pruning acts.
        amount: quantity of units to prune. A float in ``[0, 1]`` denotes
            the fraction of units to prune; an int denotes the absolute
            number of units to prune.

    Returns:
        The modified (i.e. pruned) module.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = random_unstructured(nn.Linear(2, 3), "weight", amount=1)
        >>> int(tensorplay.sum(m.weight_mask == 0))
        1
    """
    RandomUnstructured.apply(module, name, amount)
    return module


def l1_unstructured(
    module: nn.Module,
    name: str,
    amount: int | float,
    importance_scores: Tensor | None = None,
) -> nn.Module:
    """Prune ``module[name]`` by removing the units with the smallest
    magnitudes.

    Removes ``amount`` (currently unpruned) units ranked by absolute value.
    Modifies the module in place (and also returns it) by:

    1) adding a named buffer called ``name + '_mask'`` holding the binary
       mask applied to the parameter ``name``;
    2) replacing the parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Args:
        module: module containing the tensor to prune.
        name: parameter name within ``module`` on which pruning acts.
        amount: quantity of units to prune. A float in ``[0, 1]`` denotes
            the fraction of units to prune; an int denotes the absolute
            number of units to prune.
        importance_scores: tensor of importance scores with the same shape
            as the parameter; each entry ranks the corresponding element of
            the parameter. When unspecified, the parameter itself is used.

    Returns:
        The modified (i.e. pruned) module.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = l1_unstructured(nn.Linear(2, 3), "weight", amount=0.2)
        >>> list(m.state_dict().keys())
        ['bias', 'weight_orig', 'weight_mask']
    """
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


def random_structured(
    module: nn.Module, name: str, amount: int | float, dim: int
) -> nn.Module:
    """Prune ``module[name]`` by removing random channels along ``dim``.

    Removes ``amount`` (currently unpruned) channels chosen uniformly at
    random. Modifies the module in place (and also returns it) by:

    1) adding a named buffer called ``name + '_mask'`` holding the binary
       mask applied to the parameter ``name``;
    2) replacing the parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Args:
        module: module containing the tensor to prune.
        name: parameter name within ``module`` on which pruning acts.
        amount: quantity of channels to prune. A float in ``[0, 1]`` denotes
            the fraction of channels to prune; an int denotes the absolute
            number of channels to prune.
        dim: axis along which channels are defined.

    Returns:
        The modified (i.e. pruned) module.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = random_structured(nn.Linear(5, 3), "weight", amount=3, dim=1)
        >>> columns_pruned = int(sum(tensorplay.sum(m.weight, dim=0) == 0))
        >>> print(columns_pruned)
        3
    """
    RandomStructured.apply(module, name, amount, dim)
    return module


def ln_structured(
    module: nn.Module,
    name: str,
    amount: int | float,
    n: int | float | str,
    dim: int,
    importance_scores: Tensor | None = None,
) -> nn.Module:
    """Prune ``module[name]`` by removing the channels with the smallest
    L``n``-norm along ``dim``.

    Modifies the module in place (and also returns it) by:

    1) adding a named buffer called ``name + '_mask'`` holding the binary
       mask applied to the parameter ``name``;
    2) replacing the parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Args:
        module: module containing the tensor to prune.
        name: parameter name within ``module`` on which pruning acts.
        amount: quantity of channels to prune. A float in ``[0, 1]`` denotes
            the fraction of channels to prune; an int denotes the absolute
            number of channels to prune.
        n: norm order; accepts the orders valid for
            :func:`tensorplay.linalg.vector_norm` and
            :func:`tensorplay.linalg.matrix_norm`.
        dim: axis along which channels are defined.
        importance_scores: tensor of importance scores with the same shape
            as the parameter; each entry ranks the corresponding element of
            the parameter. When unspecified, the parameter itself is used.

    Returns:
        The modified (i.e. pruned) module.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = ln_structured(
        ...     nn.Conv2d(5, 3, 2), "weight", amount=0.3, dim=1, n=float("-inf")
        ... )
    """
    LnStructured.apply(
        module, name, amount, n, dim, importance_scores=importance_scores
    )
    return module


def global_unstructured(
    parameters: Iterable[tuple[nn.Module, str]],
    pruning_method: type[BasePruningMethod],
    importance_scores: dict[tuple[nn.Module, str], Tensor] | None = None,
    **kwargs: Any,
) -> None:
    """Prune several tensors jointly under a single unstructured budget.

    Aggregates the importance scores of every listed parameter into one
    vector, computes a single mask under the shared ``amount`` budget, and
    slices that mask back onto each parameter. Modifies the modules in place
    by:

    1) adding a named buffer called ``name + '_mask'`` for every listed
       parameter;
    2) replacing each parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Args:
        parameters: iterable of ``(module, name)`` tuples identifying the
            parameters to prune globally, i.e. by aggregating all values
            before deciding which units to remove.
        pruning_method: a pruning method class from this package (or a
            user-defined subclass of :class:`BasePruningMethod`) whose
            ``PRUNING_TYPE`` is ``'unstructured'``.
        importance_scores: mapping from ``(module, name)`` tuples to the
            corresponding importance-scores tensor (same shape as the
            parameter). Parameters absent from the mapping use their own
            values as importance scores.
        kwargs: keyword arguments forwarded to ``pruning_method``, typically
            ``amount``: the quantity of units to prune across all listed
            parameters (a float fraction in ``[0, 1]`` or an absolute int).

    Raises:
        TypeError: if ``parameters`` is not an iterable, if
            ``importance_scores`` is not a dict, or if the ``PRUNING_TYPE``
            of ``pruning_method`` is not ``'unstructured'``.

    Note:
        Global pruning is restricted to unstructured methods: a structured
        norm is only comparable across channels of equal size, which cannot
        be guaranteed across heterogeneous parameters.

    Examples:
        >>> # xdoctest: +SKIP
        >>> net = nn.Sequential(nn.Linear(10, 4), nn.Linear(4, 1))
        >>> parameters_to_prune = (
        ...     (net[0], "weight"),
        ...     (net[1], "weight"),
        ... )
        >>> global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=L1Unstructured,
        ...     amount=10,
        ... )
    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = tensorplay.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = tensorplay.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", tensorplay.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            f"the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}"
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        custom_from_mask(module, name, mask=param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


def custom_from_mask(module: nn.Module, name: str, mask: Tensor) -> nn.Module:
    """Prune ``module[name]`` with a pre-computed binary ``mask``.

    Modifies the module in place (and also returns it) by:

    1) adding a named buffer called ``name + '_mask'`` holding ``mask``;
    2) replacing the parameter ``name`` by its masked version, while the
       original (unmasked) values are stored in a new parameter named
       ``name + '_orig'``.

    Args:
        module: module containing the tensor to prune.
        name: parameter name within ``module`` on which pruning acts.
        mask: binary mask to be applied to the parameter.

    Returns:
        The modified (i.e. pruned) module.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = custom_from_mask(
        ...     nn.Linear(5, 3), name="bias", mask=tensorplay.tensor([0, 1, 0])
        ... )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])
    """
    CustomFromMask.apply(module, name, mask)
    return module


def remove(module: nn.Module, name: str) -> nn.Module:
    """Make the pruning of ``module[name]`` permanent and drop the
    reparameterization.

    The pruned parameter ``name`` remains permanently pruned, the parameter
    ``name + '_orig'`` is removed from the parameter list, and the buffer
    ``name + '_mask'`` is removed from the buffers. The pruning hook is also
    detached from the module.

    Note:
        Pruning itself is NOT undone or reversed!

    Args:
        module: module containing the pruned tensor.
        name: parameter name within ``module`` whose pruning is removed.

    Returns:
        The modified module.

    Raises:
        ValueError: if ``name`` is not currently pruned on ``module``.
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(
        f"Parameter '{name}' of module {module} has to be pruned before pruning can be removed"
    )


def is_pruned(module: nn.Module) -> bool:
    """Check whether ``module`` carries an active pruning reparameterization.

    Scans every submodule for forward pre-hooks that are instances of
    :class:`BasePruningMethod`.

    Args:
        module: module that is either pruned or unpruned.

    Returns:
        ``True`` when at least one submodule is pruned, ``False`` otherwise.

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = nn.Linear(5, 7)
        >>> print(is_pruned(m))
        False
        >>> random_unstructured(m, name="weight", amount=0.2)
        >>> print(is_pruned(m))
        True
    """
    for _, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, BasePruningMethod):
                return True
    return False
