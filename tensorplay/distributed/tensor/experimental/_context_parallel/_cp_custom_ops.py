from __future__ import annotations

from typing import Any

import tensorplay
from tensorplay.distributed import _functional_collectives as funcol
from tensorplay import library

__all__ = [
    "flex_cp_allgather",
    "flex_cp_allgather_backward",
]


@library.custom_op("cplib::flex_cp_allgather", mutates_args=())
def flex_cp_allgather(
    k: Any, v: Any, seq_dim: int, pg_name: Any
) -> tuple[Any, Any]:
    k = funcol.all_gather_single(k.contiguous(), seq_dim, group=pg_name)
    v = funcol.all_gather_single(v.contiguous(), seq_dim, group=pg_name)
    return funcol.wait_tensor(k), funcol.wait_tensor(v)


def _group_size(pg_name: Any) -> int:
    group = funcol._resolve_group(pg_name)
    return int(group.size())


@flex_cp_allgather.register_fake
def _flex_cp_allgather_fake(
    k: Any, v: Any, seq_dim: int, pg_name: Any
) -> tuple[Any, Any]:
    group_size = _group_size(pg_name)
    shape_k = list(k.shape)
    shape_v = list(v.shape)
    shape_k[seq_dim] *= group_size
    shape_v[seq_dim] *= group_size
    return (
        tensorplay.empty(shape_k, dtype=k.dtype, device=k.device),
        tensorplay.empty(shape_v, dtype=v.dtype, device=v.device),
    )


@library.custom_op("cplib::flex_cp_allgather_backward", mutates_args=())
def flex_cp_allgather_backward(
    grad_full_k: Any, grad_full_v: Any, seq_dim: int, pg_name: Any
) -> tuple[Any, Any]:
    grad_k = funcol.reduce_scatter_single(
        grad_full_k, "sum", seq_dim, group=pg_name
    )
    grad_v = funcol.reduce_scatter_single(
        grad_full_v, "sum", seq_dim, group=pg_name
    )
    return funcol.wait_tensor(grad_k), funcol.wait_tensor(grad_v)


@flex_cp_allgather_backward.register_fake
def _flex_cp_allgather_backward_fake(
    grad_full_k: Any, grad_full_v: Any, seq_dim: int, pg_name: Any
) -> tuple[Any, Any]:
    group_size = _group_size(pg_name)
    shape_k = list(grad_full_k.shape)
    shape_v = list(grad_full_v.shape)
    shape_k[seq_dim] //= group_size
    shape_v[seq_dim] //= group_size
    return (
        tensorplay.empty(shape_k, dtype=grad_full_k.dtype, device=grad_full_k.device),
        tensorplay.empty(shape_v, dtype=grad_full_v.dtype, device=grad_full_v.device),
    )


def _flex_cp_allgather_backward_autograd(
    ctx: Any, grad_full_k: Any, grad_full_v: Any
) -> tuple[Any, Any, None, None]:
    grad_k, grad_v = flex_cp_allgather_backward(
        grad_full_k, grad_full_v, ctx.seq_dim, ctx.pg_name
    )
    return grad_k, grad_v, None, None


def _flex_cp_allgather_setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: Any
) -> None:
    del output
    _, _, ctx.seq_dim, ctx.pg_name = inputs


flex_cp_allgather.register_autograd(
    _flex_cp_allgather_backward_autograd,
    setup_context=_flex_cp_allgather_setup_context,
)
