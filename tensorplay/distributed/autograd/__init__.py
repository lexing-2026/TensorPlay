# Ported from torch/distributed/autograd/__init__.py.
#
# torch's distributed autograd is part of the RPC framework: gradients are
# accumulated in a distributed context keyed by RPC context ids. tp ships no
# RPC runtime, so every entry point reports that requirement, matching the
# error torch raises when RPC is not initialized.
__all__ = ["backward", "get_gradients", "is_initialized"]


def _require_rpc():
    raise RuntimeError(
        "tensorplay.distributed.autograd requires tensorplay.distributed.rpc, "
        "which is not available in this build. Initialize RPC via "
        "rpc.init_rpc before using distributed autograd."
    )


def is_initialized() -> bool:
    """Whether the distributed autograd framework is initialized."""
    return False


def backward(context_id: int, tensors, *args, **kwargs) -> None:
    """Kick off distributed backward pass (requires RPC)."""
    _require_rpc()


def get_gradients(context_id: int) -> dict:
    """Retrieve gradients accumulated in the given context (requires RPC)."""
    _require_rpc()
