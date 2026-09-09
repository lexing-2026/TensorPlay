try:
    from tensorplay._C import _distributed as _native
except ImportError:
    _native = None

if _native is None:
    def is_available() -> bool:
        return False

    __all__ = ["is_available"]
else:
    # Distributed process-group surface implemented in ``distributed_core``
    # plus the store layer (C++ backends with Python shims in ``_store``).
    from tensorplay.distributed import distributed_core as _core
    from tensorplay.distributed._store import (
        FileStore,
        HashStore,
        PrefixStore,
        Store,
        StoreTimeoutError,
        TCPStore,
    )
    from tensorplay.distributed.constants import (
        default_pg_nccl_timeout,
        default_pg_timeout,
    )
    from tensorplay.distributed import config
    from tensorplay.distributed.distributed_core import *  # noqa: F401,F403

    # (DDP, DDP comm hooks, checkpoint) rely on them, so re-export the internal
    # accessors too.
    _get_default_group = _core._get_default_group
    _resolve_group = _core._resolve_group
    _rank_not_in_group = _core._rank_not_in_group
    _broadcast_coalesced = _core._broadcast_coalesced
    _compute_bucket_assignment_by_size = _core._compute_bucket_assignment_by_size
    _verify_params_across_processes = _core._verify_params_across_processes

    from tensorplay.distributed.rendezvous import (
        register_rendezvous_handler,
        rendezvous,
    )
    from tensorplay.distributed.remote_device import _remote_device
    from tensorplay.distributed import tensor as tensor

    __all__ = list(_core.__all__) + [
        "FileStore",
        "HashStore",
        "PrefixStore",
        "Store",
        "StoreTimeoutError",
        "TCPStore",
        "default_pg_timeout",
        "default_pg_nccl_timeout",
        "config",
        "register_rendezvous_handler",
        "rendezvous",
        "tensor",
    ]
