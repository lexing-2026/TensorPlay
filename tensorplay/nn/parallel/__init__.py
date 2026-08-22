from tensorplay.nn.parallel.distributed import DistributedDataParallel
from tensorplay.nn.parallel.data_parallel import (
    DataParallel,
    data_parallel,
    gather,
    parallel_apply,
    scatter,
)

__all__ = ["DataParallel", "DistributedDataParallel"]
