"""Calling a module with externally supplied parameters and buffers.

A module holds its parameters as state.  Function transforms need the
opposite: a pure function of ``(params, inputs)``.  The bridge is to swap the
caller's tensors into the module for the duration of one call and swap the
originals back afterwards, which is what :func:`_functional_call` does.
"""
import contextlib
from typing import Any, Optional

import tensorplay

from ._named_member_accessor import NamedMemberAccessor

__all__ = ["functional_call"]


def _untie_named_tensors_map(
    module: Any, parameters_and_buffers: dict[str, Any]
) -> dict[str, Any]:
    """Propagates each given value to every name tied to the same tensor.

    A module may expose one tensor under several names (weight tying).  A
    caller who supplies a value for one of those names usually means it for
    all of them, so the value is copied across the tied group.  Supplying
    *different* values for names in the same group is ambiguous and rejected.
    """
    all_named_tensors: dict[str, Any] = {}
    all_named_tensors.update(module.named_parameters(remove_duplicate=False))
    all_named_tensors.update(module.named_buffers(remove_duplicate=False))

    # Group names by the tensor object they resolve to.  Tensors are not
    # hashable by value here, so identity is the grouping key.
    tensor_to_tied_names_map: dict[int, set[str]] = {}
    for name, tensor in all_named_tensors.items():
        tensor_to_tied_names_map.setdefault(id(tensor), set()).add(name)

    tied_names_map: dict[str, set[str]] = {}
    for tied_names in tensor_to_tied_names_map.values():
        if len(tied_names) > 1:
            for tied_name in tied_names:
                tied_names_map[tied_name] = tied_names

    given_names_for_tied_tensors = {
        name for name in parameters_and_buffers if name in tied_names_map
    }
    for given_name in given_names_for_tied_tensors:
        tied_names = tied_names_map[given_name]
        given_in_group = tied_names.intersection(given_names_for_tied_tensors)
        if len(given_in_group) > 1 and len(
            {id(parameters_and_buffers[n]) for n in given_in_group}
        ) != 1:
            raise ValueError(
                f"functional_call got multiple values for keys {sorted(tied_names)}, "
                f"which are tied. Consider using tie_weights=False"
            )

    untied_parameters_and_buffers = parameters_and_buffers.copy()
    for given_name in given_names_for_tied_tensors:
        for tied_name in tied_names_map[given_name]:
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[given_name]
    return untied_parameters_and_buffers


@contextlib.contextmanager
def _reparametrize_module(
    module: Any,
    parameters_and_buffers: dict[str, Any],
    tie_weights: bool = False,
    strict: bool = False,
    stack_weights: bool = False,
):
    """Swaps ``parameters_and_buffers`` into ``module`` for the block's duration.

    The originals are restored on exit, including when the body raises.  If
    the module mutated its own ``_parameters``/``_buffers`` in place while
    reparameterized, those new values are written back into the caller's dict
    so the mutation is not silently lost.
    """
    if tie_weights:
        untied_parameters_and_buffers = _untie_named_tensors_map(
            module, parameters_and_buffers
        )
    else:
        untied_parameters_and_buffers = parameters_and_buffers

    accessor = NamedMemberAccessor(module)
    if strict:
        missing_keys, unexpected_keys = accessor.check_keys(untied_parameters_and_buffers)
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append(f"Unexpected key(s): {', '.join(map(repr, unexpected_keys))}.")
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in reparametrizing for {}:\n\t{}".format(
                    module._get_name(), "\n\t".join(error_msgs)
                )
            )

    orig_parameters_and_buffers: dict[str, Any] = {}
    try:
        orig_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            untied_parameters_and_buffers, allow_missing=True
        )
        yield
    finally:
        if stack_weights:
            # Restore in reverse insertion order, so a name swapped twice ends
            # on the value it started with.
            orig_parameters_and_buffers = dict(reversed(orig_parameters_and_buffers.items()))
        new_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            orig_parameters_and_buffers, allow_missing=True
        )
        parameters_and_buffers.update(
            {
                k: new_parameters_and_buffers[k]
                for k in parameters_and_buffers
                if k in new_parameters_and_buffers
            }
        )


def _functional_call(
    module: Any,
    parameters_and_buffers: dict[str, Any],
    args: Any = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    with _reparametrize_module(
        module, parameters_and_buffers, tie_weights=tie_weights, strict=strict
    ):
        return module(*args, **kwargs)


def functional_call(
    module: Any,
    parameters_and_buffers: dict[str, Any],
    args: Any = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    """Calls ``module`` with the given parameters and buffers in place of its own.

    :func:`tensorplay.func.functional_call` is the supported entry point and
    additionally accepts several dicts; this one exists for the stateless
    machinery itself.
    """
    return _functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )
