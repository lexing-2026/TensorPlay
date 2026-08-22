# Ported from torch/distributed/rpc/__init__.py (availability gate).
#
# torch gates the RPC API on the C++ ``_rpc_init`` symbol; tp has no RPC
# runtime, so the public names report the same "RPC not initialized"
# requirement as an uninitialized torch RPC stack.
__all__: list[str] = []


def _require_rpc():
    raise RuntimeError(
        "tensorplay.distributed.rpc requires a build with the RPC runtime "
        "(torch parity: torch._C._rpc_init); not available in this build."
    )


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    _require_rpc()
