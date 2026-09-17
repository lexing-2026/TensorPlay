"""Tests for tensorplay.package: exporting pure-Python module packages and
loading them back hermetically."""

import io
import re
import sys
import zipfile
from pathlib import Path

import pytest

import tensorplay as tp

import tensorplay.package as tp_package
from tensorplay.package import (
    EmptyMatchError,
    GlobGroup,
    PackageExporter,
    PackageImporter,
    PackagingError,
    is_from_package,
)
from tensorplay.package._mangling import (
    PackageMangler,
    demangle,
    get_mangle_prefix,
    is_mangled,
)


MANGLE_ID_RE = re.compile(r"^<tensorplay_package_\d+>$")

MATH_UTILS_SRC = '''\
def add_one(x):
    return x + 1

def scale(x, factor):
    return x * factor

THREE = 3
'''

NET_SRC = '''\
import tensorplay as tp
from my_package.math_utils import add_one


class TinyNet(tp.nn.Module):
    """A tiny module that only uses Python state in __init__ and creates
    tensors at call time."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return add_one(x * self.factor) + tp.tensor([1.0])
'''


def _write_my_package(root: Path):
    pkg = root / "my_package"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "math_utils.py").write_text(MATH_UTILS_SRC)
    (pkg / "net.py").write_text(NET_SRC)
    return pkg


@pytest.fixture
def package_env(tmp_path, monkeypatch):
    """A directory holding an importable `my_package` package, put on sys.path."""
    _write_my_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    # Purge the real modules so later tests re-import fresh copies.
    for name in list(sys.modules):
        if name == "my_package" or name.startswith("my_package."):
            del sys.modules[name]


def _export_basic(root, archive_name="pkg.zip"):
    archive = root / archive_name
    with PackageExporter(archive) as exp:
        exp.intern("my_package.**")
        exp.save_module("my_package")
        exp.save_module("my_package.math_utils")
        exp.save_module("my_package.net")
    return archive


# ---------------------------------------------------------------------------
# Basic round trip: source capture and hermetic re-import
# ---------------------------------------------------------------------------


def test_round_trip_module_behavior(package_env):
    import my_package.math_utils as math_utils

    archive = _export_basic(package_env)

    imp = PackageImporter(archive)
    imported = imp.import_module("my_package.math_utils")

    # Behavior of the packaged code matches the original.
    assert imported.add_one(41) == math_utils.add_one(41)
    assert imported.scale(3, 5) == math_utils.scale(3, 5)
    assert imported.THREE == math_utils.THREE == 3


def test_round_trip_nn_module(package_env):
    import my_package.net as orig_net

    archive = package_env / "pkg.zip"
    with PackageExporter(archive) as exp:
        exp.intern("my_package.**")
        exp.save_module("my_package")
        exp.save_pickle("my_package", "model", orig_net.TinyNet(2.0))

    imp = PackageImporter(archive)
    imported = imp.import_module("my_package.net")

    net = imported.TinyNet(2.0)
    x = 1.5
    local = orig_net.TinyNet(2.0)
    assert tp.allclose(net.forward(x), local.forward(x))


def test_save_pickle_loads_from_package_environment(package_env):
    import my_package.net as orig_net

    archive = package_env / "pkg.zip"
    with PackageExporter(archive) as exp:
        exp.intern("my_package.**")
        exp.save_module("my_package")
        exp.save_pickle("my_package", "model", orig_net.TinyNet(3.0))

    imp = PackageImporter(archive)
    loaded = imp.load_pickle("my_package", "model")

    # The unpickled object is reconstructed against the package's own copy of
    # the defining class, not the class from the local environment.
    imported = imp.import_module("my_package.net")
    assert type(loaded) is imported.TinyNet
    assert type(loaded) is not orig_net.TinyNet
    assert isinstance(loaded.factor, float) or isinstance(loaded.factor, int)

    local = orig_net.TinyNet(3.0)
    assert tp.allclose(loaded.forward(2.0), local.forward(2.0))


# ---------------------------------------------------------------------------
# Environment isolation and the mangled name scheme
# ---------------------------------------------------------------------------


def test_imported_module_is_isolated(package_env):
    import my_package.net as orig_net

    archive = _export_basic(package_env)
    imp = PackageImporter(archive)
    imported = imp.import_module("my_package.net")

    # The module is loaded under the importer's synthetic parent, not under
    # its original name, and is not registered in sys.modules.
    assert imported.__name__ != "my_package.net"
    assert f"{imp.id()}.my_package.net" == imported.__name__
    assert imported.__name__ not in sys.modules

    # Class identity differs: same source, distinct class objects.
    assert imported.TinyNet is not orig_net.TinyNet
    assert is_from_package(imported)
    assert not is_from_package(orig_net)
    # Instances built from the packaged class are recognized as packaged.
    assert is_from_package(imported.TinyNet(1.0))
    assert not is_from_package(42)


def test_mangled_name_scheme(package_env):
    archive = _export_basic(package_env)
    imp = PackageImporter(archive)

    assert MANGLE_ID_RE.match(imp.id())
    imported = imp.import_module("my_package.math_utils")
    assert imported.__name__.startswith(imp.id() + ".")
    assert imported.__name__ == f"{imp.id()}.my_package.math_utils"
    # The archive-facing name kept on the loader is not mangled.
    assert demangle(imported.__name__) == "my_package.math_utils"


def test_mangling_helpers():
    m = PackageMangler()
    mangled = m.mangle("foo.bar")
    assert mangled.startswith("<tensorplay_package_")
    assert mangled.endswith(">.foo.bar")
    assert m.demangle(mangled) == "foo.bar"
    # A different mangler's names pass through untouched.
    other = PackageMangler()
    assert other.demangle(mangled) == mangled

    assert is_mangled("<tensorplay_package_0>.foo")
    assert not is_mangled("foo")
    assert is_mangled("<tensorplay_package_12>")
    assert demangle("<tensorplay_package_0>") == ""
    assert demangle("plain.mod") == "plain.mod"
    assert get_mangle_prefix("<tensorplay_package_3>.a.b") == "<tensorplay_package_3>"
    # Non-mangled names pass through unchanged.
    assert get_mangle_prefix("a.b") == "a.b"


def test_importer_ids_are_unique():
    first = PackageMangler()
    second = PackageMangler()
    assert first.parent_name() != second.parent_name()


def test_two_importers_same_archive(package_env):
    archive = _export_basic(package_env)
    imp_a = PackageImporter(archive)
    imp_b = PackageImporter(archive)
    assert imp_a.id() != imp_b.id()

    mod_a = imp_a.import_module("my_package.math_utils")
    mod_b = imp_b.import_module("my_package.math_utils")
    assert mod_a.__name__ != mod_b.__name__
    assert mod_a.add_one is not mod_b.add_one
    assert mod_a.add_one(1) == mod_b.add_one(1) == 2


# ---------------------------------------------------------------------------
# Glob inclusion / exclusion
# ---------------------------------------------------------------------------


def test_glob_group_matching():
    gg = GlobGroup("tensorplay.**")
    assert gg.matches("tensorplay")
    assert gg.matches("tensorplay.nn")
    assert gg.matches("tensorplay.nn.functional")
    assert not gg.matches("tensorplayx.nn")

    gg = GlobGroup("tensorplay.*")
    assert gg.matches("tensorplay.nn")
    assert not gg.matches("tensorplay.nn.functional")

    gg = GlobGroup(["alpha.*", "beta.**"])
    assert gg.matches("alpha.one")
    assert gg.matches("beta.a.b.c")
    assert not gg.matches("gamma.x")

    # Exclusion removes otherwise-included candidates.
    gg = GlobGroup("my_package.**", exclude="my_package.private.**")
    assert gg.matches("my_package.net")
    assert not gg.matches("my_package.private.secrets")

    # Double wildcard must occupy an entire segment.
    with pytest.raises(ValueError):
        GlobGroup("foo**bar")


def test_glob_group_path_separator():
    gg = GlobGroup("data/**/*.txt", separator="/")
    assert gg.matches("data/sub/file.txt")
    assert gg.matches("data/a/b/file.txt")
    assert not gg.matches("data/file.dat")


def test_exporter_intern_exclude_extern(package_env):
    (package_env / "my_package" / "public_helper.py").write_text(
        "def help_out():\n    return 'helped'\n"
    )
    (package_env / "my_package" / "public_user.py").write_text(
        "from my_package import public_helper\n\n\ndef use_it():\n    return public_helper.help_out()\n"
    )
    import my_package.public_helper

    archive = package_env / "pkg.zip"
    with PackageExporter(archive) as exp:
        # Intern the package but keep one module external to the archive.
        exp.intern("my_package.**", exclude="my_package.public_helper")
        exp.extern("my_package.public_helper")
        exp.save_module("my_package")
        exp.save_module("my_package.public_user")
        exp.save_module("my_package.net")

    assert "my_package.public_helper" in exp.externed_modules()
    assert "my_package.public_helper" not in exp.interned_modules()
    assert "my_package.public_user" in exp.interned_modules()

    imp = PackageImporter(archive)
    # An externed module resolves against the surrounding interpreter.
    imported_helper = imp.import_module("my_package.public_helper")
    import sys

    assert imported_helper is sys.modules["my_package.public_helper"]
    # Interned ones come from the archive.
    imported_net = imp.import_module("my_package.net")
    assert imported_net.__name__.startswith("<tensorplay_package_")


def test_extern_allow_empty_false(package_env):
    archive = package_env / "pkg.zip"
    with pytest.raises(EmptyMatchError):
        with PackageExporter(archive) as exp:
            exp.intern("my_package.**")
            exp.extern("missing_module_name.**", allow_empty=False)
            exp.save_module("my_package")
            exp.save_module("my_package.math_utils")
            exp.save_module("my_package.net")


# ---------------------------------------------------------------------------
# deny / PackagingError
# ---------------------------------------------------------------------------


def test_deny_raises_packaging_error(package_env):
    # The denied module must be importable in the exporting environment; the
    # denial only kicks in when the packaged code requires it.
    (package_env / "disallowed_lib.py").write_text("thing = 1\n")
    (package_env / "my_package" / "bad_user.py").write_text(
        "import disallowed_lib\n\n\ndef run():\n    return disallowed_lib.thing\n"
    )
    archive = package_env / "pkg.zip"
    with pytest.raises(PackagingError) as err:
        with PackageExporter(archive) as exp:
            exp.intern("my_package.**")
            exp.deny("disallowed_lib")
            exp.save_module("my_package")
            exp.save_module("my_package.bad_user")
    assert "Module was denied by a pattern." in str(err.value)


def test_no_action_raises_packaging_error(package_env):
    (package_env / "never_listed_lib.py").write_text("x = 1\n")
    (package_env / "my_package" / "orphan_user.py").write_text(
        "import never_listed_lib\n\n\ndef run():\n    return never_listed_lib.x\n"
    )
    archive = package_env / "pkg.zip"
    with pytest.raises(PackagingError) as err:
        with PackageExporter(archive) as exp:
            exp.intern("my_package.**")
            exp.save_module("my_package")
            exp.save_module("my_package.orphan_user")
    assert "never_listed_lib" in str(err.value)


# ---------------------------------------------------------------------------
# Mocking
# ---------------------------------------------------------------------------


def test_mock_replaces_module_attributes(package_env):
    (package_env / "heavy_lib.py").write_text(
        "def heavy_compute(x):\n    return x * 1000\n"
    )
    (package_env / "my_package" / "reporter.py").write_text(
        "import heavy_lib\n\n\ndef report(x):\n    return heavy_lib.heavy_compute(x)\n"
    )
    sys.path.insert(0, str(package_env))
    try:
        archive = package_env / "pkg.zip"
        with PackageExporter(archive) as exp:
            exp.intern("my_package.**")
            exp.mock("heavy_lib")
            exp.save_module("my_package")
            exp.save_module("my_package.reporter")

        assert exp.mocked_modules() == ["heavy_lib"]

        imp = PackageImporter(archive)
        reporter = imp.import_module("my_package.reporter")
        # Attribute access on the mocked module hands out a stub object.
        assert "MockedObject" in repr(reporter.heavy_lib.heavy_compute)
        # Using a stub raises instead of executing the real code.
        with pytest.raises(NotImplementedError):
            reporter.report(2)
        assert reporter.__name__.startswith("<tensorplay_package_")
    finally:
        sys.path.remove(str(package_env))
        for name in ["heavy_lib", "my_package", "my_package.reporter"]:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Resources: text/binary save and load
# ---------------------------------------------------------------------------


def test_save_and_load_text_binary(package_env):
    archive = package_env / "pkg.zip"
    with PackageExporter(archive) as exp:
        exp.intern("my_package.**")
        exp.save_module("my_package")
        exp.save_module("my_package.math_utils")
        exp.save_module("my_package.net")
        exp.save_text("my_package", "notes.txt", "hello package")
        exp.save_binary("my_package", "blob.bin", b"\x00\x01\x02")

    imp = PackageImporter(archive)
    assert imp.load_text("my_package", "notes.txt") == "hello package"
    assert imp.load_binary("my_package", "blob.bin") == b"\x00\x01\x02"


def test_python_version_record(package_env):
    archive = _export_basic(package_env)
    imp = PackageImporter(archive)
    version = imp.python_version()
    assert version is not None
    assert version.split(".")[:2] == [str(sys.version_info[0]), str(sys.version_info[1])]


# ---------------------------------------------------------------------------
# File structure representation
# ---------------------------------------------------------------------------


def test_file_structure(package_env):
    archive = _export_basic(package_env)
    imp = PackageImporter(archive)
    directory = imp.file_structure()

    assert directory.has_file("my_package/net.py")
    assert directory.has_file("my_package/__init__.py")
    assert not directory.has_file("my_package/missing.py")

    filtered = imp.file_structure(include="my_package/net.py")
    assert filtered.has_file("my_package/net.py")
    assert not filtered.has_file("my_package/math_utils.py")


# ---------------------------------------------------------------------------
# Alternative container forms: buffer and unzipped directory
# ---------------------------------------------------------------------------


def test_export_and_import_through_buffer(package_env):
    buffer = io.BytesIO()
    with PackageExporter(buffer) as exp:
        exp.intern("my_package.**")
        exp.save_module("my_package")
        exp.save_module("my_package.math_utils")
    assert buffer.getvalue()[:2] == b"PK"  # a valid zip archive

    imp = PackageImporter(buffer)
    imported = imp.import_module("my_package.math_utils")
    assert imported.add_one(4) == 5


def test_import_from_unzipped_directory(package_env, tmp_path):
    archive = _export_basic(package_env)
    unzipped = tmp_path / "unzipped"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(unzipped)

    imp = PackageImporter(unzipped)
    imported = imp.import_module("my_package.math_utils")
    assert imported.add_one(9) == 10
    assert is_from_package(imported)


def test_exporter_source_string_and_file(package_env, tmp_path):
    archive = package_env / "pkg.zip"

    # save_source_string does not require the module to be importable locally.
    with PackageExporter(archive) as exp:
        exp.intern("standalone.**")
        exp.save_source_string("standalone", "VALUE = 7\n", is_package=True)
        exp.save_source_string("standalone.calc", "def double(v):\n    return v * 2\n")

    imp = PackageImporter(archive)
    mod = imp.import_module("standalone.calc")
    assert mod.double(21) == 42

    # save_source_file picks up a directory tree of python sources.
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "__init__.py").write_text("")
    (tree / "leaf.py").write_text("ANSWER = 10\n")
    archive2 = package_env / "pkg2.zip"
    with PackageExporter(archive2) as exp:
        exp.intern("tree.**")
        exp.save_source_file("tree", str(tree))
    imp2 = PackageImporter(archive2)
    assert imp2.import_module("tree.leaf").ANSWER == 10


# ---------------------------------------------------------------------------
# Exporter bookkeeping APIs
# ---------------------------------------------------------------------------


def test_exporter_introspection_and_hooks(package_env):
    archive = package_env / "pkg.zip"
    interned_seen = []
    externed_seen = []
    with PackageExporter(archive) as exp:
        exp.intern("my_package.**")
        handle = exp.register_intern_hook(lambda exporter, name: interned_seen.append(name))
        exp.register_extern_hook(lambda exporter, name: externed_seen.append(name))
        exp.save_module("my_package")
        exp.save_module("my_package.math_utils")
        exp.save_module("my_package.net")

    # Hooks ran while the dependency graph was executed at close() time.
    assert "my_package" in interned_seen
    assert "my_package.net" in interned_seen
    assert "tensorplay" in externed_seen
    handle.remove()

    assert "my_package" in interned_seen
    assert "my_package.net" in interned_seen
    assert "tensorplay" in externed_seen

    assert "my_package.net" in exp.interned_modules()
    assert "tensorplay" in exp.externed_modules()
    assert "my_package.net" in exp.get_rdeps("my_package.math_utils")

    graph = exp.dependency_graph_string()
    assert "my_package.net" in graph
    assert graph.startswith("digraph G {")


def test_save_module_requires_string(package_env):
    with PackageExporter(package_env / "pkg.zip") as exp:
        with pytest.raises(TypeError):
            exp.save_module(object())


def test_package_error_reports_extension_modules(package_env):
    # A C extension module cannot be interned; the failure must be reported
    # through PackagingError with a clear reason. A module whose __file__ has
    # an extension suffix models one without needing a real binary.
    import types

    fake_ext = types.ModuleType("fake_ext")
    fake_ext.__file__ = str(package_env / "fake_ext.cpython-x86_64-linux-gnu.so")
    sys.modules["fake_ext"] = fake_ext
    try:
        archive = package_env / "pkg.zip"
        with pytest.raises(PackagingError) as err:
            with PackageExporter(archive) as exp:
                exp.intern("my_package.**")
                exp.intern("fake_ext")
                exp.save_module("my_package")
                exp.save_module("my_package.math_utils")
                exp.save_module("my_package.net")
                exp.save_module("fake_ext")
        assert "C extension module" in str(err.value)
    finally:
        sys.modules.pop("fake_ext", None)


def test_debug_packaging_error_includes_path(package_env):
    (package_env / "my_package" / "bad_user.py").write_text(
        "import disallowed_lib\n"
    )
    archive = package_env / "pkg.zip"
    with pytest.raises(PackagingError) as err:
        with PackageExporter(archive, debug=True) as exp:
            exp.intern("my_package.**")
            exp.deny("disallowed_lib")
            exp.save_module("my_package")
            exp.save_module("my_package.bad_user")
    assert "A path to disallowed_lib" in str(err.value)


def test_top_level_api_exports():
    for name in [
        "PackageExporter",
        "PackageImporter",
        "PackagingError",
        "EmptyMatchError",
        "Importer",
        "OrderedImporter",
        "ObjNotFoundError",
        "ObjMismatchError",
        "GlobGroup",
        "is_from_package",
        "sys_importer",
    ]:
        assert hasattr(tp_package, name)
