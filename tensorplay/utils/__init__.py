from . import data

def __getattr__(name):
    if name == "viz":
        from . import viz
        return viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
