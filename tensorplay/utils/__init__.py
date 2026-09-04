from . import data

def __getattr__(name):
    if name == "viz":
        from . import viz
        return viz
    if name == "tensorboard":
        from . import tensorboard
        return tensorboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
