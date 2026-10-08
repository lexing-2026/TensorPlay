import sys

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="CUDA wheel preload is Linux-only")


def test_missing_nvjpeg_preloads_it(monkeypatch):
    # _C links nvjpeg for the image IO bindings, so a wheel installed next
    # to the NVIDIA runtime wheels fails on it like on any other CUDA lib.
    requested = []
    monkeypatch.setattr(
        tp, "_preload_cuda_lib",
        lambda folder, name, required=True: requested.append((folder, name, required)))
    tp._preload_cuda_deps(ImportError(
        "libnvjpeg.so.13: cannot open shared object file: No such file or directory"))
    assert ("nvjpeg", "libnvjpeg.so.*[0-9]", False) in requested


def test_unrelated_import_error_is_reraised():
    err = ImportError("libfoo.so.1: cannot open shared object file")
    with pytest.raises(ImportError, match="libfoo"):
        tp._preload_cuda_deps(err)
