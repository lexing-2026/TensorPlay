import importlib
import os
import shutil
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

import tensorplay
from tensorplay import hub


@pytest.fixture(autouse=True)
def _isolate_hub_dir(monkeypatch):
    """Keep tests away from the real hub cache on disk."""
    monkeypatch.setattr(hub, "_hub_dir", None)
    monkeypatch.delenv(hub.ENV_HOME, raising=False)
    monkeypatch.delenv(hub.ENV_XDG_CACHE_HOME, raising=False)


def _write_repo(directory, entrypoints="answer", dependencies=None):
    lines = []
    if dependencies is not None:
        lines.append(f"dependencies = {dependencies!r}")
    lines.append("def _helper():\n    return 1")
    lines.append(
        "def answer(a=1, b=2):\n    '''sum a and b'''\n    return a + b"
    )
    (directory / "hubconf.py").write_text("\n".join(lines) + "\n")


def test_get_dir_resolves_env(tmp_path, monkeypatch):
    monkeypatch.setenv(hub.ENV_HOME, str(tmp_path))
    assert hub.get_dir() == tmp_path / "hub"

    monkeypatch.delenv(hub.ENV_HOME)
    monkeypatch.setenv(hub.ENV_XDG_CACHE_HOME, str(tmp_path))
    assert hub.get_dir() == tmp_path / "tensorplay" / "hub"

    monkeypatch.delenv(hub.ENV_XDG_CACHE_HOME)
    assert hub.get_dir() == Path.home() / ".cache" / "tensorplay" / "hub"


def test_set_dir_overrides(monkeypatch, tmp_path):
    hub.set_dir(tmp_path / "custom")
    assert hub.get_dir() == tmp_path / "custom"


def test_parse_repo_info_default_main(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(hub, "urlopen", lambda *a, **k: _Resp())
    assert hub._parse_repo_info("owner/repo") == ("owner", "repo", "main")


def test_parse_repo_info_default_master(monkeypatch):
    def _raise(*a, **k):
        raise HTTPError("http://x", 404, "not found", None, None)

    monkeypatch.setattr(hub, "urlopen", _raise)
    assert hub._parse_repo_info("owner/repo") == ("owner", "repo", "master")


def test_list_filters_helpers_and_dependencies(tmp_path, monkeypatch):
    _write_repo(tmp_path, dependencies=["tensorplay"])
    monkeypatch.setattr(hub, "_get_cache_or_reload", lambda *a, **k: tmp_path)
    assert hub.list("owner/repo") == ["answer"]


def test_help_returns_docstring(tmp_path, monkeypatch):
    _write_repo(tmp_path)
    monkeypatch.setattr(hub, "_get_cache_or_reload", lambda *a, **k: tmp_path)
    assert "sum a and b" in hub.help("owner/repo", "answer")


def test_load_from_directory(tmp_path):
    _write_repo(tmp_path)
    assert hub.load(str(tmp_path), "answer", 1, b=2, source="local") == 3


def test_load_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown source"):
        hub.load(str(tmp_path), "answer", source="ftp")


def test_missing_dependencies_raise(tmp_path):
    _write_repo(tmp_path, dependencies=["no_such_package_xyz_123"])
    with pytest.raises(RuntimeError, match="Missing dependencies"):
        hub.load(str(tmp_path), "answer", source="local")


def test_safe_extract_rejects_traversal(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as zf:
        zf.writestr("../../evil.txt", "boom")
    with pytest.raises(ValueError, match="traversal|escape"):
        with zipfile.ZipFile(tmp_path / "evil.zip") as zf:
            hub._safe_extract_zip(zf, tmp_path)


def test_cache_and_force_reload(monkeypatch, tmp_path):
    hub.set_dir(tmp_path / "hub")
    calls = {"n": 0}

    def fake_download(url, dst, **kwargs):
        calls["n"] += 1
        with zipfile.ZipFile(dst, "w") as zf:
            zf.writestr("repo-abc/hubconf.py", "x = 1")

    monkeypatch.setattr(hub, "download_url_to_file", fake_download)
    monkeypatch.setattr(hub, "_validate_ref", lambda *a: None)

    repo_dir = hub._get_cache_or_reload(
        "owner/repo:main", force_reload=True, trust_repo=True
    )
    assert (repo_dir / "hubconf.py").exists()
    assert calls["n"] == 1

    cached = hub._get_cache_or_reload(
        "owner/repo:main", force_reload=False, trust_repo=True
    )
    assert cached == repo_dir
    assert calls["n"] == 1  # cache hit, no second download

    hub._get_cache_or_reload("owner/repo:main", force_reload=True, trust_repo=True)
    assert calls["n"] == 2  # force_reload re-downloads


def test_load_state_dict_from_url(monkeypatch, tmp_path):
    source = tmp_path / "src.pth"
    tensorplay.save({"w": tensorplay.tensor([1.0, 2.0])}, source)

    def fake_download(url, dst, **kwargs):
        shutil.copyfile(source, dst)

    monkeypatch.setattr(hub, "download_url_to_file", fake_download)
    model_dir = tmp_path / "ckpts"
    loaded = hub.load_state_dict_from_url(
        "https://example.com/weights-0123456789abcdef.pth",
        model_dir=model_dir,
        file_name="custom.pth",
        check_hash=True,
        progress=False,
    )
    assert (model_dir / "custom.pth").exists()
    assert "w" in loaded
