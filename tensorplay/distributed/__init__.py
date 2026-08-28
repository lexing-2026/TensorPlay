# Ported from torch/distributed/__init__.py: thin facade re-exporting the
# c10d surface implemented in ``distributed_c10d`` plus the pure-Python
# store layer.
from tensorplay.distributed import distributed_c10d as _c10d
from tensorplay.distributed._store import FileStore, Store, StoreTimeoutError, TCPStore
from tensorplay.distributed.constants import (
    default_pg_nccl_timeout,
    default_pg_timeout,
)
from tensorplay.distributed.distributed_c10d import *  # noqa: F401,F403

# Private-but-de-facto-public names: torch exposes
# ``torch.distributed._get_default_group`` and friends, and tp call sites
# (DDP, DDP comm hooks, checkpoint) rely on them, so re-export the internal
# accessors too.
_get_default_group = _c10d._get_default_group
_resolve_group = _c10d._resolve_group
_rank_not_in_group = _c10d._rank_not_in_group
_broadcast_coalesced = _c10d._broadcast_coalesced
_compute_bucket_assignment_by_size = _c10d._compute_bucket_assignment_by_size
_verify_params_across_processes = _c10d._verify_params_across_processes

from tensorplay.distributed.rendezvous import (
    register_rendezvous_handler,
    rendezvous,
)

__all__ = list(_c10d.__all__) + [
    "FileStore",
    "Store",
    "StoreTimeoutError",
    "TCPStore",
    "default_pg_timeout",
    "default_pg_nccl_timeout",
    "register_rendezvous_handler",
    "rendezvous",
]
