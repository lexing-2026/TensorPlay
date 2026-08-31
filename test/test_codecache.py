"""L5-M1: content-addressed kernel cache behaviour."""

import os

import pytest

from tensorplay._stax import CodeCache, default_cache


def test_cache_key_canonicalization(tmp_path):
    cache = CodeCache("triton", root=str(tmp_path))
    k1 = cache.cache_key("SRC", "kernel", {"b": 2, "a": 1})
    k2 = cache.cache_key("SRC", "kernel", {"a": 1, "b": 2})
    assert k1 == k2
    assert cache.cache_key("SRC2", "kernel", {"a": 1}) != k1
    assert cache.cache_key("SRC", "other", {"a": 1}) != k1
    other = CodeCache("avx2", root=str(tmp_path))
    assert other.cache_key("SRC", "kernel", {"a": 1}) != k1


def test_compile_or_load_calls_once(tmp_path):
    cache = CodeCache("testbk", root=str(tmp_path))
    calls = []

    def compile_fn(src):
        calls.append(src)
        return b"BIN-" + src.encode()

    src = "kernel_body_v1"
    a1, p1 = cache.compile_or_load(compile_fn, src, entry="k", ext="cubin")
    a2, p2 = cache.compile_or_load(compile_fn, src, entry="k", ext="cubin")
    assert calls == [src]
    assert a1 == a2 == b"BIN-kernel_body_v1"
    assert p1 == p2 and os.path.exists(p1)


def test_store_is_atomic_and_loadable(tmp_path):
    cache = CodeCache("bk", root=str(tmp_path))
    key = cache.cache_key("s")
    path = cache.store(key, b"payload", ext="so")
    with open(path, "rb") as fh:
        assert fh.read() == b"payload"
    leftovers = [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")]
    assert leftovers == []


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path / "envroot"))
    cache = default_cache("envbk")
    assert str(tmp_path / "envroot") in cache.root
