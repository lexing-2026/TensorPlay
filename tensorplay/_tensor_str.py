# mypy: allow-untyped-defs
"""Tensor printing options.

Rendering itself lives in the native layer; this module tracks the
requested options so the current configuration can be queried and
temporarily overridden. Everything set through :func:`set_printoptions`
is forwarded to the native layer.
"""

from contextlib import contextmanager

from tensorplay._C import set_printoptions as _native_set_printoptions

__all__ = ["set_printoptions", "get_printoptions", "printoptions"]

_DEFAULTS = {
    "precision": 4,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 80,
}

_current = dict(_DEFAULTS)


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    edge_items=None,
    **kwargs,
):
    """Set printing options for tensor output.

    Args:
        precision (int, optional): digits after the decimal point.
        threshold (int, optional): total elements above which large
            tensors are summarized instead of printed fully.
        edgeitems (int, optional): edge items printed per dimension
            when summarizing. ``edge_items`` is accepted as an alias.
        linewidth (int, optional): line width for wrapping output.

    Unset (``None``) arguments keep their current values.
    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(
            f"set_printoptions() got unexpected keyword(s): {names}; "
            "supported keywords are precision, threshold, edgeitems, "
            "linewidth"
        )
    if edgeitems is not None and edge_items is not None:
        raise TypeError(
            "set_printoptions() got both 'edgeitems' and 'edge_items'; "
            "pass only one"
        )
    if edge_items is None:
        edge_items = edgeitems
    for name, value in (
        ("precision", precision),
        ("threshold", threshold),
        ("edgeitems", edge_items),
        ("linewidth", linewidth),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"set_printoptions() argument '{name}' must be int, "
                    f"not {type(value).__name__}"
                )
            _current[name] = value
    _native_set_printoptions(
        edge_items=-1 if edge_items is None else edge_items,
        threshold=-1 if threshold is None else threshold,
        precision=-1 if precision is None else precision,
        linewidth=-1 if linewidth is None else linewidth,
    )


def get_printoptions() -> dict:
    """Current tensor printing options.

    The values reflect the defaults until :func:`set_printoptions`
    overrides them.
    """
    return dict(_current)


@contextmanager
def printoptions(**kwargs):
    """Temporarily set printing options within a ``with`` block."""
    previous = dict(_current)
    try:
        set_printoptions(**kwargs)
        yield
    finally:
        set_printoptions(**previous)
