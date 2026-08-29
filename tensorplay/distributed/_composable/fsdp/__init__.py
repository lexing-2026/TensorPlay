#
# fully_shard requires the DTensor stack (tensor/) for sharded parameter
# placement; tracked in docs/gap_analysis.md.
__all__: list[str] = []


def __getattr__(name):
    if name == "fully_shard":
        raise NotImplementedError(
            "fully_shard requires the DTensor stack "
            "(tensorplay.distributed.tensor), which is not yet implemented."
        )
    raise AttributeError(name)
