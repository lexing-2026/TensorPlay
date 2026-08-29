#
# Adaptation: tp's DDP has no side-stream upcast machinery; the hook
# performs the reduce-precision allreduce and upcast numerics inline.
from dataclasses import dataclass
from typing import Any, no_type_check

import tensorplay as tp

import tensorplay.distributed as dist
import tensorplay.futures


@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.

    This contains a weakref to the DDP module for access to reducer and process
    """

    ddp_weakref: Any
    upcast_stream: Any = None
    wait_for_stream_enqueued: bool = False


@no_type_check
def _reducer_allreduce_and_upcast_hook(
    hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket
):
    """
    Perform allreduce in precision ``reduce_dtype``, upcast to prepare for optimizer.

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    """
    ddp_weakref = hook_state.ddp_weakref
    process_group = ddp_weakref().process_group
    # Cast bucket if different than param_dtype.
    if (
        ddp_weakref().mixed_precision.param_dtype
        != ddp_weakref().mixed_precision.reduce_dtype
    ):
        # Cast bucket tensor to reduce_dtype
        bucket.set_buffer(
            bucket.buffer().to(ddp_weakref().mixed_precision.reduce_dtype)
        )
    buffer = bucket.buffer()
    work = dist.all_reduce(buffer, group=process_group, async_op=True)
    ret_fut = tp.futures.Future()

    work.wait()
    buffer.div_(process_group.size())

    # Upcast parameters and gradients so optimizer step can run in fp32.
    for p in bucket.parameters():
        p.grad.data = p.grad.data.to(tp.float32)

    ret_fut.set_result(bucket.buffer())
    return ret_fut
