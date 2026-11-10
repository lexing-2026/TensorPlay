"""The out-of-tree backend path, end to end, through the openreg reference
backend (test/cpp_extensions/openreg).

The backend is compiled against the p10 headers with
tensorplay.utils.cpp_extension and exercised in a child process (see
openreg/scenario.py for why); each scenario case is reported as its own
test here.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

tp = pytest.importorskip("tensorplay")

HERE = Path(__file__).resolve().parent
SCENARIO = HERE / "openreg" / "scenario.py"

CASES = [
    "naming_before_load",
    "naming_after_rename",
    "factories_and_copies",
    "views_need_no_kernels",
    "arithmetic_matches_host",
    "autograd_through_backend_kernels",
    "dispatch_keys_and_abi",
    "accelerator_and_device_module",
    "generated_methods",
    "custom_op_kernel_for_the_device",
    "autocast_on_the_device",
    "vmap_over_device_tensors",
    "missing_kernel_is_reported",
]

pytestmark = pytest.mark.skipif(
    not (shutil.which(os.environ.get("CXX", "g++")) or shutil.which("c++")),
    reason="no C++ compiler to build the reference backend",
)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    env = dict(os.environ)
    env["OPENREG_BUILD_DIR"] = str(tmp_path_factory.mktemp("openreg_build"))
    env["TENSORPLAY_DEVICE_BACKEND_AUTOLOAD"] = "0"
    proc = subprocess.run(
        [sys.executable, str(SCENARIO)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        pytest.fail(f"scenario crashed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return json.loads(lines[-1])


@pytest.mark.parametrize("name", CASES)
def test_openreg(results, name):
    if "load_backend" in results:
        pytest.fail("building or loading the backend failed:\n" + results["load_backend"])
    assert name in results, f"case {name} did not run; ran: {sorted(results)}"
    assert results[name] == "ok", results[name]


def test_every_case_is_listed():
    sys.path.insert(0, str(SCENARIO.parent))
    try:
        import scenario  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    assert sorted(scenario.CASES) == sorted(CASES)


def test_key_families_cover_every_backend():
    # Every backend component, the out-of-tree one included, owns one key in
    # each layered family; the names come from the key enumeration itself.
    names = set(tp._C.DispatchKey.__members__)
    backends = tp._C._dispatch_get_backend_keyset_from_composite()
    assert "PrivateUse1" in backends
    for backend in backends:
        for prefix in ("Autograd", "Autocast", "Vmap"):
            assert prefix + backend in names, prefix + backend
            key = tp._C._dispatch_key_parse(prefix + backend)
            assert tp._C._dispatch_key_name(tp._C._dispatch_key_to_backend_key(key)) == backend


def test_unnamed_slot_spelling_round_trips():
    dev = tp.device("privateuseone:2")
    assert (dev.type, dev.index, str(dev)) == ("privateuseone", 2, "privateuseone:2")
    assert tp.device(str(dev)) == dev
    for spelling in ("xpu", "hip", "hpu", "mtia"):
        assert tp.device(spelling).type == spelling
