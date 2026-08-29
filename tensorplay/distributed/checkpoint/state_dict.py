"""State-dict helpers for plain modules, DDP, and optimizers.

The supported subset covers the plain-module and DDP paths of get_state_dict /
set_state_dict / get_model_state_dict / set_model_state_dict /
get_optimizer_state_dict / set_optimizer_state_dict. FSDP/DTensor-specific
options are accepted-and-ignored or raise, matching what tp can honor.
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import tensorplay as tp

__all__ = [
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "get_state_dict",
    "set_model_state_dict",
    "set_optimizer_state_dict",
    "set_state_dict",
]


def _unwrap(model):
    return getattr(model, "module", model)


def get_model_state_dict(
    model,
    *,
    options=None,
) -> Dict[str, Any]:
    inner = _unwrap(model)
    sd = {k: (v.detach().clone() if isinstance(v, tp.Tensor) else v)
          for k, v in inner.state_dict().items()}
    return sd


def set_model_state_dict(
    model,
    model_state_dict,
    *,
    options=None,
) -> None:
    inner = _unwrap(model)
    inner.load_state_dict(model_state_dict)


def get_optimizer_state_dict(
    model,
    optimizers,
    *,
    options=None,
) -> Dict[str, Any]:
    optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
    out = {"state": {}, "param_groups": []}
    for opt in optimizers:
        osd = opt.state_dict()
        out["state"].update(osd.get("state", {}))
        out["param_groups"].extend(osd.get("param_groups", []))
    return out


def set_optimizer_state_dict(
    model,
    optimizers,
    *,
    optimizer_state_dict,
    options=None,
) -> None:
    optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
    for opt in optimizers:
        opt.load_state_dict(optimizer_state_dict)


def get_state_dict(model, optimizers, *, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return get_model_state_dict(model), get_optimizer_state_dict(model, optimizers)


def set_state_dict(
    model,
    optimizers,
    *,
    model_state_dict,
    optimizer_state_dict,
    options=None,
) -> None:
    set_model_state_dict(model, model_state_dict)
    set_optimizer_state_dict(model, optimizers, optimizer_state_dict=optimizer_state_dict)
