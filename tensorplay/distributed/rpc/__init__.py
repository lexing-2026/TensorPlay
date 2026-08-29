#
# runtime, so the public names report the same "RPC not initialized"
__all__: list[str] = []


def _require_rpc():
    raise RuntimeError(
        "tensorplay.distributed.rpc requires a build with the RPC runtime "
    )


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    _require_rpc()
