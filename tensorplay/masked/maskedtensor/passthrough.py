# mypy: allow-untyped-defs
"""
These are functions that should simply be applied to both mask and data.
Take select or stack as an example. This operation can be applied to
both the mask and data of a MaskedTensor and the result wrapped into
a new MaskedTensor as a result.
"""

import tensorplay as tp

from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


# Names are resolved against the module-level function first and the tensor
# method second; names this codebase does not provide are skipped, since the
# corresponding operation is unreachable through the explicit application
# entry point anyway.
_PASSTHROUGH_NAMES = [
    "select",
    "transpose",
    "split",
    "t",
    "slice",
    "slice_backward",
    "select_backward",
    "index",
    "expand",
    "view",
    "_unsafe_view",
    "_reshape_alias",
    "cat",
    "unsqueeze",
    "unfold",
    "unfold_backward",
    "im2col",
    "col2im",
    "stack",
]


def _resolve_passthrough_fns():
    """Resolve passthrough operations to (table key, applying callable) pairs.

    The table key is the module-level function, which is hashable; the
    applying callable is the tensor method, which forwards shape-style
    positional arguments correctly.
    """
    fns = []
    for name in _PASSTHROUGH_NAMES:
        key = getattr(tp, name, None)
        applier = getattr(tp.Tensor, name, None)
        if key is None:
            if applier is None:
                continue
            key = applier
        if applier is None:
            applier = key
        fns.append((key, applier))
    return fns


PASSTHROUGH_KEYS = [key for key, _ in _resolve_passthrough_fns()]
_APPLIERS = {key: applier for key, applier in _resolve_passthrough_fns()}
PASSTHROUGH_FNS = PASSTHROUGH_KEYS


def _is_pass_through_fn(fn):
    return fn in PASSTHROUGH_FNS


def _apply_pass_through_fn(fn, *args, **kwargs):
    applier = _APPLIERS.get(fn, fn)
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    result_data = applier(*data_args, **data_kwargs)
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())
    result_mask = applier(*mask_args, **mask_kwargs)
    return _wrap_result(result_data, result_mask)
