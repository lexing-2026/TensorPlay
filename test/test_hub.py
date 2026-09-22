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


def test_collect_weight_paths_prefers_mega_index(tmp_path):
    (tmp_path / "model-00001-of-00001.mega").write_bytes(b"x")
    (tmp_path / "model.mega.index.json").write_text("{}")
    (tmp_path / "README.md").write_text("x")
    paths = hub._collect_weight_paths(tmp_path)
    assert paths == [tmp_path / "model.mega.index.json"]


def test_collect_weight_paths_extensions(tmp_path):
    (tmp_path / "w.safetensors").write_bytes(b"x")
    (tmp_path / "w.bin").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x")
    paths = hub._collect_weight_paths(tmp_path)
    assert sorted(p.name for p in paths) == ["w.bin", "w.safetensors"]


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


# ---------------------------------------------------------------------------
# Foreign-hubconf compatibility: hubconf files written against foreign module
# names load on top of the native packages.
# ---------------------------------------------------------------------------

_FOREIGN_HUBCONF = """\
dependencies = ["torch"]

from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.extension import _HAS_OPS
from torchvision.models import get_model_weights
from torchvision.models.alexnet import alexnet
from torchvision.models.optical_flow import Raft_Large_Weights


def build_alexnet():
    return alexnet()


def nested_import():
    from torchvision.models.resnet import resnet18
    return resnet18


def uses_tensor():
    return Tensor([1.0]).sum().item()


def ops_available():
    return bool(_HAS_OPS)
"""


@pytest.fixture
def foreign_repo(tmp_path):
    (tmp_path / "hubconf.py").write_text(_FOREIGN_HUBCONF)
    return tmp_path


def test_foreign_dependencies_satisfied(foreign_repo):
    # "torch" resolves through the compat mapping, so the dependency check passes
    # even without that package installed.
    assert hub.load(str(foreign_repo), "ops_available", source="local") is True


def test_foreign_imports_map_to_native_modules(foreign_repo):
    model = hub.load(str(foreign_repo), "build_alexnet", source="local")
    assert type(model).__module__ == "tensorplay.vision.models.alexnet"


def test_foreign_nested_import_at_entry_time(foreign_repo):
    resnet18 = hub.load(str(foreign_repo), "nested_import", source="local")()
    assert type(resnet18).__module__ == "tensorplay.vision.models.resnet"


def test_foreign_tensor_identity(foreign_repo):
    assert hub.load(str(foreign_repo), "uses_tensor", source="local") == 1.0


def test_foreign_aliases_point_at_native_modules(foreign_repo):
    import sys

    hub.load(str(foreign_repo), "build_alexnet", source="local")
    alias_keys = [
        k for k in sys.modules if k == "torch" or k.startswith(("torch.", "torchvision"))
    ]
    assert "torch" in alias_keys and "torchvision.models.alexnet" in alias_keys
    for key in alias_keys:
        # Aliased entries are the native modules themselves, never a foreign
        # package installed in the environment.
        assert sys.modules[key].__name__.startswith("tensorplay"), key
