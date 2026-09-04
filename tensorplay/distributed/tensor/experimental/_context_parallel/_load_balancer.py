"""Sequence index balancers for context-parallel attention workloads."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any

import tensorplay as tp
from tensorplay.func import vmap

__all__ = [
    "_HeadTailLoadBalancer",
    "_LoadBalancer",
    "_PerDocumentHeadTailLoadBalancer",
    "_PTRRLoadBalancer",
]


class _LoadBalancer(ABC):
    @abstractmethod
    def _generate_indices(self, restore: bool = False) -> tp.Tensor | None:
        raise NotImplementedError


class _HeadTailLoadBalancer(_LoadBalancer):
    def __init__(self, seq_length: int, world_size: int, device: Any) -> None:
        self.seq_length = seq_length
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> tp.Tensor:
        seq_length = self.seq_length
        world_size = self.world_size
        if seq_length % (world_size * 2) != 0:
            raise AssertionError
        chunk_size = seq_length // (world_size * 2)
        indices = tp.arange(seq_length, dtype=tp.int, device=self.device)
        chunks = indices.view(world_size * 2, chunk_size)
        head_idx = tp.arange(world_size, device=self.device)
        tail_idx = 2 * world_size - 1 - head_idx
        paired = tp.stack([chunks[head_idx], chunks[tail_idx]], dim=1)
        all_indices_tensor = paired.reshape(-1)
        if restore:
            all_indices_tensor = tp.argsort(all_indices_tensor)

        return all_indices_tensor.unsqueeze(0)


class _PerDocumentHeadTailLoadBalancer(_LoadBalancer):
    def __init__(
        self,
        seq_length_per_doc: list[list[int]],
        world_size: int,
        device: Any,
    ) -> None:
        self.seq_length_per_doc = seq_length_per_doc
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> tp.Tensor:
        return tp.stack(
            [
                self._generate_indices_for_batch(seq_lengths, restore)
                for seq_lengths in self.seq_length_per_doc
            ]
        )

    def _generate_indices_for_batch(
        self, seq_length_per_doc: list[int], restore: bool
    ) -> tp.Tensor:
        world_size = self.world_size
        device = self.device
        if not all(
            seq_length % (2 * world_size) == 0
            for seq_length in seq_length_per_doc
        ):
            raise AssertionError
        head_idx = tp.arange(world_size, device=device)
        tail_idx = 2 * world_size - 1 - head_idx
        per_doc_rank_chunks = []
        document_start_idx = 0
        for seq_length in seq_length_per_doc:
            chunk_length = seq_length // (2 * world_size)
            chunks = tp.arange(
                document_start_idx,
                document_start_idx + seq_length,
                device=device,
            ).view(2 * world_size, chunk_length)
            paired = tp.stack([chunks[head_idx], chunks[tail_idx]], dim=1)
            per_doc_rank_chunks.append(paired.reshape(world_size, -1))
            document_start_idx += seq_length
        indices_tensor = tp.cat(per_doc_rank_chunks, dim=1).reshape(-1)
        if restore:
            indices_tensor = tp.argsort(indices_tensor)

        return indices_tensor


class _PTRRLoadBalancer(_LoadBalancer):
    def __init__(self, block_mask: Any, world_size: int) -> None:
        self.block_mask = block_mask
        self.world_size = world_size

    @staticmethod
    def ptrr_scheduling(process_time: tp.Tensor, group_size: int) -> tp.Tensor:
        if process_time.ndim != 1:
            raise AssertionError
        num_tasks = process_time.size(0)

        if num_tasks % group_size != 0:
            raise NotImplementedError(
                f"num_tasks {num_tasks} must be divisible by group_size {group_size}"
            )
        device = process_time.device
        _, sorted_indices_descending = tp.sort(process_time, descending=True)
        sorted_indices_descending_reversed = tp.flip(
            sorted_indices_descending.view(-1, group_size), dims=[1]
        ).view(-1)
        tasks_in_group = tp.where(
            tp.arange(num_tasks, device=device) // group_size % 2 == 0,
            sorted_indices_descending,
            sorted_indices_descending_reversed,
        )
        tasks_in_group = tasks_in_group.view(-1, group_size).transpose(0, 1)
        tasks_in_group, _ = tp.sort(tasks_in_group, dim=1)
        return tasks_in_group

    def _generate_indices(self, restore: bool = False) -> tp.Tensor:
        block_mask = self.block_mask
        kv_num_blocks = block_mask.kv_num_blocks
        full_kv_num_blocks = block_mask.full_kv_num_blocks
        non_sparse_kv_num_blocks = (
            kv_num_blocks + full_kv_num_blocks
            if full_kv_num_blocks is not None
            else kv_num_blocks
        )
        B, _, Q = non_sparse_kv_num_blocks.shape
        non_sparse_kv_num_blocks = non_sparse_kv_num_blocks.view(-1, Q)
        batch_ptrr = vmap(
            functools.partial(
                _PTRRLoadBalancer.ptrr_scheduling,
                group_size=self.world_size,
            )
        )
        ptrr_indices = batch_ptrr(non_sparse_kv_num_blocks)
        ptrr_indices = ptrr_indices.reshape(B, -1)
        q_blk_size, kv_blk_size = block_mask.BLOCK_SIZE
        if q_blk_size != kv_blk_size:
            raise AssertionError("for now only support q_blk_size == kv_blk_size")

        indices = tp.arange(
            q_blk_size * ptrr_indices.size(1), device=ptrr_indices.device
        ).view(-1, q_blk_size)
        indices = indices[ptrr_indices].view(B, -1)
        if restore:
            indices = vmap(tp.argsort)(indices)

        return indices


def _create_default_load_balancer(
    seq_length: int, world_size: int, device: Any
) -> _LoadBalancer | None:
    from ._attention import _cp_options

    if _cp_options.enable_load_balance and seq_length % (world_size * 2) == 0:
        return _HeadTailLoadBalancer(seq_length, world_size, device)
    else:
        return None
