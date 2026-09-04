"""Calling modules as pure functions of their state.

A module holds its parameters; a transform needs a function of them.  These
helpers bridge the two: :func:`functional_call` runs a module with a supplied
state dict instead of its own, and :func:`stack_module_state` turns a list of
identically-shaped modules into one stacked state that :func:`~tensorplay.func.vmap`
can map over -- an ensemble evaluated as a single batched call.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay
import tensorplay.nn as nn
from tensorplay.nn.utils._named_member_accessor import NamedMemberAccessor

from .utils import exposed_in

__all__ = ["functional_call", "stack_module_state", "construct_stacked_leaf"]


@exposed_in("tensorplay.func")
def functional_call(
    module: "nn.Module",
    parameter_and_buffer_dicts: Union[dict[str, Any], Sequence[dict[str, Any]]],
    args: Optional[Union[Any, tuple]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    """Runs ``module`` with the parameters and buffers given, not its own.

    The module's own state is put back afterwards, including if ``module``
    raises, so this is safe to call on a live model.

    Args:
        module (tensorplay.nn.Module): the module to call.
        parameter_and_buffer_dicts (dict or sequence of dicts): the state to
            substitute, keyed by the names ``named_parameters`` and
            ``named_buffers`` report.  Several dicts are merged; overlapping
            keys are an error, since which one wins would be arbitrary.
        args (Any or tuple): positional arguments for the module.  A non-tuple
            value is passed as the single argument.
        kwargs (dict): keyword arguments for the module.
        tie_weights (bool): when the module ties two names to one tensor, keep
            them tied by requiring the replacement to be shared as well.
            Default: ``True``.
        strict (bool): reject names that the module does not have.  Default:
            ``False``.

    Example:

        >>> params = dict(model.named_parameters())
        >>> functional_call(model, params, (x,))
    """
    if isinstance(parameter_and_buffer_dicts, dict):
        parameters_and_buffers = parameter_and_buffer_dicts
    elif isinstance(parameter_and_buffer_dicts, Sequence) and not isinstance(
        parameter_and_buffer_dicts, (str, bytes)
    ):
        if not all(isinstance(d, dict) for d in parameter_and_buffer_dicts):
            raise ValueError(
                "Expected all elements of parameter_and_buffer_dicts to be dictionaries"
            )
        all_keys = [k for d in parameter_and_buffer_dicts for k in d.keys()]
        repeated_keys = [key for key, n in Counter(all_keys).items() if n > 1]
        if len(repeated_keys) > 0:
            raise ValueError(
                f"{repeated_keys} appeared in multiple dictionaries; behavior of "
                "functional call is ambiguous"
            )
        parameters_and_buffers = {
            k: v for d in parameter_and_buffer_dicts for k, v in d.items()
        }
    else:
        raise ValueError(
            "Expected parameter_and_buffer_dicts to be a dict, or a list/tuple of "
            f"dicts, but got {type(parameter_and_buffer_dicts)}"
        )

    return nn.utils.stateless._functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )


@exposed_in("tensorplay.func")
def stack_module_state(
    models: list["nn.Module"],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stacks the state of several identical modules into batched tensors.

    Pair the result with :func:`functional_call` under
    :func:`~tensorplay.func.vmap` to evaluate a whole ensemble in one call
    instead of looping over the models.

    All models must be the same class and in the same training mode -- a mix
    would make the stacked call mean two different things at once.

    Returns:
        ``(stacked_params, stacked_buffers)``, each keyed as the modules'
        ``named_parameters``/``named_buffers`` are, with a new leading
        dimension of length ``len(models)``.

    Example:

        >>> params, buffers = stack_module_state(models)
        >>> def call(p, b, x):
        ...     return functional_call(base_model, (p, b), (x,))
        >>> vmap(call)(params, buffers, batched_x)
    """
    if len(models) == 0:
        raise RuntimeError("stack_module_state: Expected at least one model, got 0.")
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same training/eval mode."
        )
    model0_typ = type(models[0])
    if not all(type(m) is model0_typ for m in models):
        raise RuntimeError(
            "stack_module_state: Expected all models to be of the same class."
        )

    final_params: dict[str, Any] = {}
    final_buffers: dict[str, Any] = {}
    all_params = [dict(model.named_parameters()) for model in models]
    all_buffers = [dict(model.named_buffers()) for model in models]

    keys = all_params[0].keys()
    if any(entry.keys() != keys for entry in all_params[1:]):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same parameter names."
        )
    for name in keys:
        final_params[name] = construct_stacked_leaf(
            tuple(entry[name] for entry in all_params), name
        )

    keys = all_buffers[0].keys()
    if any(entry.keys() != keys for entry in all_buffers[1:]):
        raise RuntimeError(
            "stack_module_state: Expected all models to have the same buffer names."
        )
    for name in keys:
        final_buffers[name] = construct_stacked_leaf(
            tuple(entry[name] for entry in all_buffers), name
        )

    return final_params, final_buffers


def construct_stacked_leaf(
    tensors: Union[tuple["tensorplay.Tensor", ...], list["tensorplay.Tensor"]],
    name: str,
) -> "tensorplay.Tensor":
    """Stacks one named tensor across models into a single batched leaf."""
    all_requires_grad = all(t.requires_grad for t in tensors)
    none_requires_grad = all(not t.requires_grad for t in tensors)
    if not all_requires_grad and not none_requires_grad:
        raise RuntimeError(
            f"Expected {name} from each model to have the same .requires_grad"
        )
    result = tensorplay.stack(list(tensors))
    if all_requires_grad:
        # The stack is the leaf now; the per-model tensors are not its inputs.
        result = result.detach().requires_grad_()
    return result
