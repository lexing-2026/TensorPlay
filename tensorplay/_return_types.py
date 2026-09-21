"""

public contract exposes named fields (``values``/``indices``). Namedtuples are
tuple subclasses so existing positional unpacking keeps working.

``.sum()``, ...) is forwarded to the ``values`` component, so
``m = t.max(dim=0); m.shape == m.values.shape``.

The remaining multi-output reductions (cummax/cummin/kthvalue/mode/median/
nanmedian/sort/topk/aminmax) follow the same convention; ``aminmax`` is the
one with distinct field names (``min``/``max``). All of them are collected in
``all_return_types`` so tooling can discover every multi-output shape.
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

cummax_return_type = collections.namedtuple("cummax_return_type", ["values", "indices"])
cummin_return_type = collections.namedtuple("cummin_return_type", ["values", "indices"])
kthvalue_return_type = collections.namedtuple("kthvalue_return_type", ["values", "indices"])
median_return_type = collections.namedtuple("median_return_type", ["values", "indices"])
mode_return_type = collections.namedtuple("mode_return_type", ["values", "indices"])
nanmedian_return_type = collections.namedtuple("nanmedian_return_type", ["values", "indices"])
sort_return_type = collections.namedtuple("sort_return_type", ["values", "indices"])
topk_return_type = collections.namedtuple("topk_return_type", ["values", "indices"])
aminmax_return_type = collections.namedtuple("aminmax_return_type", ["min", "max"])

all_return_types = [
    max_return_type,
    min_return_type,
    cummax_return_type,
    cummin_return_type,
    kthvalue_return_type,
    median_return_type,
    mode_return_type,
    nanmedian_return_type,
    sort_return_type,
    topk_return_type,
    aminmax_return_type,
]
