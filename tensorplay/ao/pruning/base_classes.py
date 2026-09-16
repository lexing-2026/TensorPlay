"""Base classes shared by every pruning method.

A pruning method reparameterizes one named tensor of a module: the original
values move to a parameter called ``<name>_orig``, a binary mask is stored in
a buffer called ``<name>_mask``, and the attribute ``<name>`` holds their
elementwise product. A forward pre-hook recomputes that product before every
forward pass, so gradients keep flowing into the original values while the
masked entries stay at zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import tensorplay
from tensorplay import Tensor, nn

__all__ = ["PruningBase", "BasePruningMethod", "PruningContainer"]


class BasePruningMethod(ABC):
    """Abstract base class for pruning techniques.

    Subclasses must override :meth:`compute_mask` and declare a
    ``PRUNING_TYPE`` class attribute (one of ``'unstructured'``,
    ``'structured'`` or ``'global'``). The classmethod :meth:`apply` installs
    the reparameterization on a module and registers an instance of the
    subclass as a forward pre-hook.
    """

    _tensor_name: str

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        """Recompute the pruned tensor as the mask times the original values.

        Called as a forward pre-hook: multiplies the mask stored in
        ``module[name + '_mask']`` into the original values stored in
        ``module[name + '_orig']`` and stores the product into
        ``module[name]``.

        Args:
            module: module holding the pruned tensor.
            inputs: unused; present for hook-signature compatibility.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """Compute the pruning mask for the input tensor ``t``.

        Starting from ``default_mask`` (a mask of ones when ``t`` has never
        been pruned), derive the new mask according to the recipe of the
        concrete method. Entries already zeroed by ``default_mask`` must stay
        zero.

        Args:
            t: tensor whose entries or channels are ranked for pruning,
                typically the importance scores of the parameter.
            default_mask: mask accumulated from previous pruning iterations;
                must be respected by the new mask. Same shape as ``t``.

        Returns:
            The mask to apply to ``t``, with the same shape as ``t``.
        """

    def apply_mask(self, module: nn.Module) -> Tensor:
        """Return the pruned version of the tensor held by ``module``.

        Fetches the mask and the original values from the module and returns
        their elementwise product.

        Args:
            module: module holding the pruned tensor.

        Returns:
            The product of the mask and the original values.
        """
        # the mask can only be multiplied once it has been computed, so the
        # method must know which tensor it operates on; this attribute is set
        # by apply()
        if self._tensor_name is None:
            raise AssertionError(
                f"Module {module} has to be pruned"
            )  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(
        cls,
        module: nn.Module,
        name: str,
        *args: Any,
        importance_scores: Tensor | None = None,
        **kwargs: Any,
    ) -> "BasePruningMethod":
        """Install the pruning reparameterization for ``module[name]``.

        Moves the original parameter to ``name + '_orig'``, registers the
        computed mask as the buffer ``name + '_mask'``, stores the masked
        values under ``name`` and adds a forward pre-hook that re-applies the
        mask on every forward pass. If the tensor is already pruned by
        another method, the new method is composed with the existing one
        through a :class:`PruningContainer`.

        Args:
            module: module containing the tensor to prune.
            name: parameter name within ``module`` on which pruning acts.
            args: positional arguments forwarded to the subclass constructor.
            importance_scores: tensor of importance scores with the same
                shape as ``module[name]``; each entry ranks the corresponding
                element of the parameter. When unspecified, the parameter
                itself is used as its own importance scores.
            kwargs: keyword arguments forwarded to the subclass constructor.

        Returns:
            The pruning method (or container of methods) now attached to the
            module.
        """

        def _get_composite_method(
            cls, module: nn.Module, name: str, *args: Any, **kwargs: Any
        ) -> "BasePruningMethod":
            # Check if a pruning method has already been applied to
            # `module[name]`. If so, store that in `old_method`.
            old_method = None
            found = 0
            # there should technically be only 1 hook with hook._tensor_name
            # == name; assert this using `found`
            hooks_to_remove = []
            for k, hook in module._forward_pre_hooks.items():
                # if it exists, take existing thing, remove hook, then
                # go through normal thing
                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    hooks_to_remove.append(k)
                    found += 1
            if found > 1:
                raise AssertionError(
                    f"Avoid adding multiple pruning hooks to the "
                    f"same tensor {name} of module {module}. Use a PruningContainer."
                )

            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]

            # Apply the new pruning method, either from scratch or on top of
            # the previous one.
            method = cls(*args, **kwargs)  # new pruning
            # Have the pruning method remember what tensor it's been applied to
            method._tensor_name = name

            # combine `methods` with `old_method`, if `old_method` exists
            if old_method is not None:  # meaning that there was a hook
                # if the hook is already a pruning container, just add the
                # new pruning method to the container
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method  # rename old_method --> method

                # if the hook is simply a single pruning method, create a
                # container, add the old pruning method and the new one
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    container.add_pruning_method(method)
                    method = container  # rename container --> method
            return method

        method = _get_composite_method(cls, module, name, *args, **kwargs)
        # at this point we have no forward_pre_hooks but we could have an
        # active reparameterization of the tensor if another pruning method
        # had been applied (in which case `method` would be a PruningContainer
        # and not a simple pruning method).

        # Pruning is to be applied to the module's tensor named `name`,
        # starting from the state it is found in prior to this iteration of
        # pruning. The pruning mask is calculated based on importance scores.

        orig = getattr(module, name)
        if importance_scores is not None:
            if importance_scores.shape != orig.shape:
                raise AssertionError(
                    f"importance_scores should have the same shape as parameter "
                    f"{name} of {module}, got {importance_scores.shape} vs {orig.shape}"
                )
        else:
            importance_scores = orig

        # If this is the first time pruning is applied, take care of moving
        # the original tensor to a new parameter called name + '_orig'
        # and deleting the original parameter
        if not isinstance(method, PruningContainer):
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(name + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[name]
            default_mask = tensorplay.ones_like(orig)  # temp
        # If this is not the first time pruning is applied, all of the above
        # has been done before in a previous pruning iteration, so we're good
        # to go
        else:
            default_mask = (
                getattr(module, name + "_mask")
                .detach()
                .clone(memory_format=tensorplay.contiguous_format)
            )

        # Use try/except because if anything goes wrong with the mask
        # computation etc., you'd want to roll back.
        try:
            # get the final mask, computed according to the specific method
            mask = method.compute_mask(importance_scores, default_mask=default_mask)
            # reparameterize by saving mask to `module[name + '_mask']`...
            module.register_buffer(name + "_mask", mask)
            # ... and the new pruned tensor to `module[name]`
            setattr(module, name, method.apply_mask(module))
            # associate the pruning method to the module via a hook to
            # compute the function before every forward() (compile by run)
            module.register_forward_pre_hook(method)

        except Exception as e:
            if not isinstance(method, PruningContainer):
                orig = getattr(module, name + "_orig")
                module.register_parameter(name, orig)
                del module._parameters[name + "_orig"]
            raise e

        return method

    def prune(
        self,
        t: Tensor,
        default_mask: Tensor | None = None,
        importance_scores: Tensor | None = None,
    ) -> Tensor:
        """Return a pruned copy of the input tensor ``t``.

        Applies the rule implemented by :meth:`compute_mask` without any
        module-side reparameterization.

        Args:
            t: tensor to prune (same shape as ``default_mask``).
            importance_scores: tensor of importance scores with the same
                shape as ``t``; each entry ranks the corresponding element of
                ``t``. When unspecified, ``t`` itself is used.
            default_mask: mask from a previous pruning iteration, if any.
                Pruning must respect the entries it already zeroes. When
                unspecified, a mask of ones is used.

        Returns:
            The pruned version of ``t``.
        """
        if importance_scores is not None:
            if importance_scores.shape != t.shape:
                raise AssertionError(
                    f"importance_scores should have the same shape as tensor t, "
                    f"got {importance_scores.shape} vs {t.shape}"
                )
        else:
            importance_scores = t
        default_mask = default_mask if default_mask is not None else tensorplay.ones_like(t)
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module: nn.Module) -> None:
        """Make the current pruning of ``module`` permanent.

        The pruned values remain pruned: the product of mask and original
        values is written back into the parameter ``name``, and the auxiliary
        parameter ``name + '_orig'`` and buffer ``name + '_mask'`` are
        dropped.

        Note:
            Pruning itself is NOT undone or reversed!
        """
        # before removing pruning from a tensor, it has to have been applied
        if self._tensor_name is None:
            raise AssertionError(
                f"Module {module} has to be pruned before pruning can be removed"
            )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.apply_mask(module)  # masked weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]
        setattr(module, self._tensor_name, orig)


class PruningContainer(BasePruningMethod):
    """Sequence of pruning methods applied iteratively to the same tensor.

    Tracks the order in which methods were added and combines successive
    pruning calls: each new method only ranks the entries or channels that
    the previous masks left unpruned.

    Accepts as argument an instance of a :class:`BasePruningMethod` or an
    iterable of them.
    """

    def __init__(self, *args: BasePruningMethod) -> None:
        self._pruning_methods: tuple[BasePruningMethod, ...] = ()
        if not isinstance(args, Iterable):  # only 1 item
            self._tensor_name = args._tensor_name  # type: ignore[attr-defined]
            self.add_pruning_method(args)  # type: ignore[arg-type]

        elif len(args) == 1:  # only 1 item in a tuple
            self._tensor_name = args[0]._tensor_name

            self.add_pruning_method(args[0])
        else:  # manual construction from list or other iterable (or no args)
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method: BasePruningMethod | None) -> None:
        """Add a child pruning ``method`` to the container.

        Args:
            method: child pruning method to be added to the container.

        Raises:
            TypeError: if ``method`` is neither ``None`` nor a
                :class:`BasePruningMethod` instance.
            ValueError: if ``method`` acts on a different tensor name than
                the methods already held by the container.
        """
        # check that we're adding a pruning method to the container
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(f"{type(method)} is not a BasePruningMethod subclass")
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(
                "Can only add pruning methods acting on "
                f"the parameter named '{self._tensor_name}' to PruningContainer {self}."
                + f" Found '{method._tensor_name}'"
            )
        # if all checks passed, add to _pruning_methods tuple
        self._pruning_methods += (method,)  # type: ignore[operator]

    def __len__(self) -> int:
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        return self._pruning_methods[idx]

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """Apply the latest method and merge its mask into ``default_mask``.

        The new partial mask is computed on the entries or channels that
        ``default_mask`` has not zeroed out. Which portion of ``t`` the new
        mask is derived from depends on the ``PRUNING_TYPE`` of the last
        method:

        * ``'unstructured'``: the mask is computed from the flattened list of
          entries not yet masked;
        * ``'structured'``: the mask is computed from the channels that still
          hold at least one unmasked entry;
        * ``'global'``: the mask is computed across all entries.

        Args:
            t: tensor representing the parameter to prune (same shape as
                ``default_mask``).
            default_mask: mask accumulated from previous pruning iterations.

        Returns:
            The mask combining the effects of ``default_mask`` and of the
            latest method, with the same shape as ``default_mask`` and ``t``.
        """

        def _combine_masks(method: BasePruningMethod, t: Tensor, mask: Tensor) -> Tensor:
            """Combine the mask of one method with a pre-existing ``mask``.

            Args:
                method: pruning method currently being applied.
                t: tensor representing the parameter to prune (same shape as
                    ``mask``).
                mask: mask accumulated from previous pruning iterations.

            Returns:
                The mask combining the effects of the old mask and of the
                current pruning method (same shape as ``mask`` and ``t``).
            """
            new_mask = mask  # start off from existing mask
            new_mask = new_mask.to(dtype=t.dtype)

            # compute a slice of t onto which the new pruning method will operate
            if method.PRUNING_TYPE == "unstructured":
                # prune entries of t where the mask is 1
                slc = mask == 1

            # for struct pruning, exclude channels that have already been
            # entirely pruned
            elif method.PRUNING_TYPE == "structured":
                if not hasattr(method, "dim"):
                    raise AttributeError(
                        "Pruning methods of PRUNING_TYPE "
                        '"structured" need to have the attribute `dim` defined.'
                    )

                # find the channels to keep by removing the ones that have been
                # zeroed out already (i.e. where sum(entries) == 0)
                n_dims = t.dim()  # "is this a 2D tensor? 3D? ..."
                dim = method.dim
                # convert negative indexing
                if dim < 0:
                    dim = n_dims + dim
                # if dim is still negative after subtracting it from n_dims
                if dim < 0:
                    raise IndexError(
                        f"Index is out of bounds for tensor with dimensions {n_dims}"
                    )
                # find channels along dim = dim that aren't already tots 0ed out
                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
                # create slice to identify what to prune
                slc: Any = [slice(None)] * n_dims
                slc[dim] = keep_channel

            elif method.PRUNING_TYPE == "global":
                n_dims = len(t.shape)  # "is this a 2D tensor? 3D? ..."
                slc = [slice(None)] * n_dims

            else:
                raise ValueError(f"Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}")

            # compute the new mask on the unpruned slice of the tensor t
            if isinstance(slc, list):
                slc = tuple(slc)
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)

            return new_mask

        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask


#: Alias of :class:`BasePruningMethod` kept for naming convenience.
PruningBase = BasePruningMethod
