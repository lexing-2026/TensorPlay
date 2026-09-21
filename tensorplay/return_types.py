"""Named-tuple containers for the multi-output operations.

Every operation that returns several tensors hands back one of these types,
so the components are reachable by name (``result.values``) as well as by
position. ``all_return_types`` collects the container classes themselves.
"""
from tensorplay._return_types import (
    aminmax_return_type,
    all_return_types,
    cummax_return_type,
    cummin_return_type,
    kthvalue_return_type,
    max_return_type,
    median_return_type,
    min_return_type,
    mode_return_type,
    nanmedian_return_type,
    sort_return_type,
    topk_return_type,
)
from tensorplay.linalg import (
    CholeskyExResult as linalg_cholesky_ex,
    EigResult as linalg_eig,
    EighResult as linalg_eigh,
    LstsqResult as linalg_lstsq,
    QRResult as linalg_qr,
    SlogdetResult as linalg_slogdet,
    SVDResult as linalg_svd,
)

__all__ = ["all_return_types"]

aminmax = aminmax_return_type
cummax = cummax_return_type
cummin = cummin_return_type
kthvalue = kthvalue_return_type
max = max_return_type
median = median_return_type
min = min_return_type
mode = mode_return_type
nanmedian = nanmedian_return_type
sort = sort_return_type
topk = topk_return_type

__all__ += [
    "aminmax",
    "cummax",
    "cummin",
    "kthvalue",
    "max",
    "median",
    "min",
    "mode",
    "nanmedian",
    "sort",
    "topk",
    "aminmax_return_type",
    "cummax_return_type",
    "cummin_return_type",
    "kthvalue_return_type",
    "max_return_type",
    "median_return_type",
    "min_return_type",
    "mode_return_type",
    "nanmedian_return_type",
    "sort_return_type",
    "topk_return_type",
    "linalg_cholesky_ex",
    "linalg_eig",
    "linalg_eigh",
    "linalg_lstsq",
    "linalg_qr",
    "linalg_slogdet",
    "linalg_svd",
]

all_return_types = all_return_types + [
    linalg_cholesky_ex,
    linalg_eig,
    linalg_eigh,
    linalg_lstsq,
    linalg_qr,
    linalg_slogdet,
    linalg_svd,
]
