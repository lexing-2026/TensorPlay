"""CUDA index_select / index_fill_ / index_copy_ row-slice handling.

Rows whose payload divides into 16-byte chunks move through the chunked slice
kernels, which need both ends of every row aligned; odd row widths and views
whose base pointer is offset by part of an element fall back to the scalar
kernels.  The cases here cover both routes on unique indices.
"""

import pytest
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import tensorplay as tp
import tensorplay.functional as F

pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="requires CUDA build of tensorplay"
)


def as_tp(t):
    return tp.from_dlpack(to_dlpack(t))


def as_torch(t):
    return from_dlpack(tp.to_dlpack(t))


def test_chunked_rows_select_fill_copy():
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(65536, 128, generator=g, device="cuda")
    idx = torch.randperm(65536, device="cuda")[:8192]

    got = as_torch(F.index_select(as_tp(x), 0, as_tp(idx)))
    assert torch.equal(got, torch.index_select(x, 0, idx))

    ty = as_tp(x.clone())
    F.index_fill_(ty, 0, as_tp(idx), 2.5)
    assert torch.equal(
        as_torch(ty), torch.index_fill(x.clone(), 0, idx, 2.5))

    src = torch.randn(8192, 128, generator=g, device="cuda")
    ty = as_tp(x.clone())
    F.index_copy_(ty, 0, as_tp(idx), as_tp(src))
    assert torch.equal(
        as_torch(ty), torch.index_copy(x.clone(), 0, idx, src))


def test_odd_row_width_select_fill():
    # 400-byte rows: not a multiple of the 16-byte chunk, so the scalar
    # kernels run.
    g = torch.Generator(device="cuda").manual_seed(12)
    x = torch.randn(300, 100, generator=g, device="cuda")
    idx = torch.randperm(300, device="cuda")[:200]

    got = as_torch(F.index_select(as_tp(x), 0, as_tp(idx)))
    assert torch.equal(got, torch.index_select(x, 0, idx))

    ty = as_tp(x.clone())
    F.index_fill_(ty, 0, as_tp(idx), -1.25)
    assert torch.equal(
        as_torch(ty), torch.index_fill(x.clone(), 0, idx, -1.25))


def test_misaligned_base_select_fill():
    # The base pointer sits one element into the allocation while the rows
    # keep 16-byte alignment, so the chunked path must decline.
    g = torch.Generator(device="cuda").manual_seed(13)
    storage = torch.randn(1 + 300 * 128, generator=g, device="cuda")
    view = storage[1:].view(300, 128)
    idx = torch.randperm(300, device="cuda")[:250]

    got = as_torch(F.index_select(as_tp(view), 0, as_tp(idx)))
    assert torch.equal(got, torch.index_select(view, 0, idx))

    tv = as_tp(view.clone())
    F.index_fill_(tv, 0, as_tp(idx), 7.0)
    assert torch.equal(
        as_torch(tv), torch.index_fill(view.clone(), 0, idx, 7.0))


def test_small_element_types_select():
    # One-byte and eight-byte elements both land in 16-byte chunks when the
    # row width allows it; 100 one-byte columns do not.
    g = torch.Generator(device="cuda").manual_seed(14)
    u8 = torch.randint(0, 256, (300, 112), generator=g, device="cuda")
    idx = torch.randperm(300, device="cuda")[:180]
    got = as_torch(F.index_select(as_tp(u8), 0, as_tp(idx)))
    assert torch.equal(got, torch.index_select(u8, 0, idx))

    f64 = torch.randn(300, 64, generator=g, device="cuda", dtype=torch.float64)
    col = torch.randperm(64, device="cuda")[:48]
    got = as_torch(F.index_select(as_tp(f64), 1, as_tp(col)))
    assert torch.equal(got, torch.index_select(f64, 1, col))


def test_non_contiguous_fill_keeps_out_of_place_path():
    g = torch.Generator(device="cuda").manual_seed(15)
    base = torch.randn(256, 256, generator=g, device="cuda")
    x = base.t()  # column-major view: the in-place kernel must not run
    idx = torch.randperm(256, device="cuda")[:128]

    tx = as_tp(base.clone()).t()
    F.index_fill_(tx, 0, as_tp(idx), 3.0)
    assert torch.equal(as_torch(tx), torch.index_fill(x.clone(), 0, idx, 3.0))


if __name__ == "__main__":
    test_chunked_rows_select_fill_copy()
    test_odd_row_width_select_fill()
    test_misaligned_base_select_fill()
    test_small_element_types_select()
    test_non_contiguous_fill_keeps_out_of_place_path()
    print("ALL OK")
