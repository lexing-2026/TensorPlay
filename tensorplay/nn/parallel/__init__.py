from tensorplay.nn.parallel.data_parallel import (
    DataParallel,
    data_parallel,
    gather,
    parallel_apply,
    scatter,
)
import tensorplay.distributed as _distributed

if _distributed.is_available():
    from tensorplay.nn.parallel.distributed import DistributedDataParallel
else:
    DistributedDataParallel = None

__all__ = ["DataParallel", "DistributedDataParallel"]
