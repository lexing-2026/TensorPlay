
import contextlib
from collections import defaultdict
from typing import Any, NamedTuple

from tensorplay import Tensor


class _GroupSwapinInfo(NamedTuple):
    live_group: dict[str, Any]
    swapin_group: dict[str, Any]
    swapin_params: list[Any]


def _validate_state_field(state: Any) -> dict[Any, Any]:
    if not isinstance(state, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires "
            "swapin_optim_state['state'] to be a dict mapping packed "
            f"parameter ids to per-param state dicts, got {type(state).__name__}."
        )
    if any(isinstance(key, Tensor) for key in state):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if any(
        isinstance(key, int) and not isinstance(value, dict)
        for key, value in state.items()
    ):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires per-parameter optimizer "
            "state entries to be dictionaries."
        )
    return state


def _validate_param_groups_field(optimizer, param_groups: Any) -> list[dict[str, Any]]:
    if not isinstance(param_groups, list):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires "
            "swapin_optim_state['param_groups'] to be a list of param-group dicts, "
            f"got {type(param_groups).__name__}."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "swapin_optim_state has a different number of parameter groups than "
            "the live optimizer."
        )
    return param_groups


def _validate_group_against_live(
    idx: int, group: dict[str, Any], swapin_group: Any
) -> list[int]:
    if not isinstance(swapin_group, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires each optimizer param group "
            "to be a dictionary."
        )
    swapin_param_ids = swapin_group.get("params")
    if not isinstance(swapin_param_ids, list) or not all(
        isinstance(pid, int) for pid in swapin_param_ids
    ):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "param_groups[*]['params'] entries keyed by packed parameter ids."
        )
    if len(group["params"]) != len(swapin_param_ids):
        raise RuntimeError(
            f"swapin_optim_state param group {idx} has a different number of "
            "params than the live optimizer param group."
        )
    swapin_only = [key for key in swapin_group if key not in group]
    live_only = [key for key in group if key not in swapin_group]
    if swapin_only or live_only:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "param group keys to exactly match the live optimizer group keys for "
            f"group {idx}. Keys only in swap-in: {swapin_only}. "
            f"Keys only in live: {live_only}."
        )
    return swapin_param_ids


def _prepare_swap_in(optimizer, swapin_parameters, swapin_optim_state):
    if not optimizer.state:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires initialized optimizer state."
        )
    if not isinstance(swapin_optim_state, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires a DCP-style optimizer state_dict."
        )
    swapin_state = _validate_state_field(swapin_optim_state.get("state"))
    swapin_param_groups = _validate_param_groups_field(
        optimizer, swapin_optim_state.get("param_groups")
    )

    flat_parameters = list(swapin_parameters.values())
    flat_param_offset = 0
    seen_param_ids: set[int] = set()
    group_swapin_infos = []
    for idx, (group, swapin_group) in enumerate(
        zip(optimizer.param_groups, swapin_param_groups, strict=True)
    ):
        swapin_param_ids = _validate_group_against_live(idx, group, swapin_group)
        seen_param_ids.update(swapin_param_ids)
        next_offset = flat_param_offset + len(swapin_param_ids)
        if next_offset > len(flat_parameters):
            raise RuntimeError(
                "swap_in_optimizer_params_and_state requires the explicit parameter "
                "state to match optimizer.param_groups ordering."
            )
        swapin_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset
        group_swapin_infos.append(
            _GroupSwapinInfo(group, swapin_group, swapin_params)
        )

    extra_keys = [key for key in swapin_state if key not in seen_param_ids]
    if extra_keys:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['state'] "
            "to be keyed only by packed parameter ids from param_groups[*]['params']; "
            f"got extra keys {extra_keys!r}."
        )
    return swapin_state, group_swapin_infos


@contextlib.contextmanager
def swap_in_optimizer_params_and_state(
    optimizer, swapin_parameters: dict[str, Any], swapin_optim_state: dict[str, Any]
):
    """Temporarily install replacement parameters and packed optimizer state."""
    state, group_swapin_infos = _prepare_swap_in(
        optimizer, swapin_parameters, swapin_optim_state
    )
    original_state = optimizer.state
    original_group_snapshots = [dict(group) for group in optimizer.param_groups]
    try:
        swapin_state = defaultdict(dict)
        for info in group_swapin_infos:
            info.live_group["params"] = info.swapin_params
            for key, value in info.swapin_group.items():
                if key != "params":
                    info.live_group[key] = value
            for swapin_param, param_id in zip(
                info.swapin_params, info.swapin_group["params"], strict=True
            ):
                swapin_state[swapin_param] = dict(state.get(param_id, {}))
        optimizer.state = swapin_state
        yield
    finally:
        for group, snapshot in zip(
            optimizer.param_groups, original_group_snapshots, strict=True
        ):
            group.clear()
            group.update(snapshot)
        optimizer.state = original_state


__all__ = ["swap_in_optimizer_params_and_state"]
