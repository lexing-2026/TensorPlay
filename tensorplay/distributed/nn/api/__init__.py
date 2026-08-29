# RPC-backed functional wrappers. All entry points require the RPC runtime.
__all__: list[str] = []


def __getattr__(name):
    raise RuntimeError(
        f"tensorplay.distributed.nn.api.{name} requires "
        "tensorplay.distributed.rpc, which is not available in this build."
    )
