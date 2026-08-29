"""

public contract exposes named fields (``values``/``indices``). Namedtuples are
tuple subclasses so existing positional unpacking keeps working.

``.sum()``, ...) is forwarded to the ``values`` component, so
``m = t.max(dim=0); m.shape == m.values.shape``.
"""

import collections


class _reduction_return_base(collections.namedtuple(
        "_base", ["values", "indices"])):

    __slots__ = ()

    def __getattr__(self, name):
        # Only reached for attributes the namedtuple itself lacks.
        return getattr(self.values, name)


max_return_type = type("max_return_type", (_reduction_return_base,),
                       {"__module__": __name__})
min_return_type = type("min_return_type", (_reduction_return_base,),
                       {"__module__": __name__})
