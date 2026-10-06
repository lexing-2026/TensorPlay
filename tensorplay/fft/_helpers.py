"""Argument spelling shared by the frequency-domain transforms.

The transforms themselves are native ops; these helpers only translate the
frontend's optional arguments into the spellings their schemas take.
"""
import operator

__all__ = ["int_list", "norm_arg", "signal_length"]


def norm_arg(norm):
    """``None`` selects the default ``"backward"`` scaling."""
    return "backward" if norm is None else norm


def signal_length(n):
    """``None`` keeps the input's length along the transformed dimension."""
    return -1 if n is None else operator.index(n)


def int_list(value):
    """An optional integer or sequence of integers, as a list (or ``None``)."""
    if value is None:
        return None
    try:
        return [operator.index(value)]
    except TypeError:
        return [operator.index(v) for v in value]
