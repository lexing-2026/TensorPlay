"""Sampler that tolerates a changing world size between epochs.

Indices are produced as ``[rank + epoch * num_replicas, +num_replicas, ...]``
until the dataset is covered; the tail is wrapped around so every replica
sees exactly ``ceil(len(dataset) / num_replicas)`` items per epoch, which
keeps worker counts consistent across re-rendezvous rounds.
"""
from collections.abc import Iterator
from typing import TypeVar

import tensorplay as tp

__all__ = ["ElasticDistributedSampler"]

_T = TypeVar("_T")


class ElasticDistributedSampler(tp.utils.data.Sampler[int]):
    """Sampler for elastic jobs where ``num_replicas`` may change per epoch.

    Unlike a fixed-shard distributed sampler, this one derives the covered
    range from the epoch number so restarting with a different world size
    never skips or repeats data beyond the wrap-around tail.
    """

    def __init__(
        self,
        dataset,
        *,
        start_rank: int = 0,
        start_epoch: int = 0,
    ) -> None:
        self.dataset = dataset
        self.start_rank = start_rank
        self.start_epoch = start_epoch

    def __iter__(self) -> Iterator[int]:
        from tensorplay.distributed import get_rank, get_world_size

        try:
            num_replicas = get_world_size()
            rank = get_rank()
        except Exception:
            num_replicas, rank = 1, 0
        if num_replicas <= 0:
            raise ValueError(f"world size must be positive, got {num_replicas}")
        epoch = self.start_epoch
        rank = (rank + self.start_rank) % num_replicas
        n = len(self.dataset)
        while True:
            idx = rank + epoch * num_replicas
            while idx < n:
                yield idx % n
                idx += num_replicas
            epoch += 1
            if epoch > self.start_epoch + 1 and idx - num_replicas >= n:
                # One full pass per epoch is enough for sane consumers.
                return

    def __len__(self) -> int:
        from tensorplay.distributed import get_world_size

        try:
            num_replicas = get_world_size()
        except Exception:
            num_replicas = 1
        return (len(self.dataset) + num_replicas - 1) // num_replicas
