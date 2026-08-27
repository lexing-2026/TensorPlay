"""End-to-end tests for the compile-time (C++) custom-operator path:
yaml -> tools/codegen/tensorplaygen.py -> g++ extension linked against
libp10 + libtp_python (single-copy bridge) -> dispatcher-backed calls.

Mirrors the add_tensorplay_op() flow from cmake/TensorPlayCustomOp.cmake,
but compiles directly so the test does not need a second CMake binary dir.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN_TOOL = REPO / "tools" / "codegen" / "tensorplaygen.py"
SAMPLE_YAML = REPO / "test" / "cpp_extension" / "sample_ops.yaml"
IMPL_CPP = REPO / "test" / "cpp_extension" / "ops_impl.cpp"
GEN_INCLUDE = REPO / "build" / "generated"

tp = pytest.importorskip("tensorplay")

pytestmark = pytest.mark.skipif(
    not (shutil.which("g++") and GEN_TOOL.exists()
         and SAMPLE_YAML.exists() and IMPL_CPP.exists()),
    reason="g++ or codegen inputs unavailable",
)


def _pybind_includes():
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pybind11", "--includes"],
            capture_output=True, text=True, check=True)
        return out.stdout.split()
    except Exception:
        pytest.skip("pybind11 headers unavailable")


@pytest.fixture(scope="module")
def myext(tmp_path_factory):
    """Build the sample `myext` extension once for the whole module."""
    import sysconfig

    gen_dir = tmp_path_factory.mktemp("gen")
    subprocess.run(
        [sys.executable, str(GEN_TOOL), "--yaml", str(SAMPLE_YAML),
         "--out_dir", str(gen_dir), "--module_name", "myext"],
        check=True, capture_output=True)

    includes = [
        str(REPO / "p10" / "include"),
        str(GEN_INCLUDE),
        str(REPO / "src" / "bindings" / "python"),
        str(gen_dir),
        sysconfig.get_paths()["include"],
    ] + _pybind_includes()

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_so = tmp_path_factory.mktemp("bin") / f"myext{suffix}"
    cmd = ["g++", "-std=c++20", "-O2", "-fPIC", "-shared"]
    cmd += [f"-I{p}" for p in includes]
    cmd += [str(gen_dir / "OpsBinding.cpp"), str(IMPL_CPP)]
    lib_dir = REPO / "tensorplay" / "lib"
    if (lib_dir / "libtp_python.so").exists():
        cmd += ["-L", str(lib_dir), "-lp10", "-ltp_python"]
    else:
        # Fallback: no prebuilt bridge library -- embed the bridge source.
        cmd.insert(-1, str(REPO / "src" / "bindings" / "python"
                           / "CPythonBridge.cpp"))
        cmd += ["-L", str(lib_dir), "-lp10"]
    cmd += ["-Wl,-rpath," + str(lib_dir), "-o", str(out_so)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"g++ failed:\n{proc.stderr[-3000:]}"

    sys.path.insert(0, str(out_so.parent))
    try:
        yield __import__("myext")
    finally:
        sys.path.remove(str(out_so.parent))
        sys.modules.pop("myext", None)


class TestCppCustomOp:
    def test_value_return(self, myext):
        x = tp.randn(3, 4)
        assert tp.allclose(myext.scale_add(x, 2.0), x * 2.0)

    def test_kwargs_and_default_scalar(self, myext):
        x = tp.randn(5)
        assert tp.allclose(myext.scale_add(x, 3.0, 1.5), x * 3.0 + 1.5)
        assert tp.allclose(myext.scale_add(factor=0.5, self=x), x * 0.5)

    def test_inplace_write_through(self, myext):
        x = tp.randn(4)
        z = x.clone()
        myext.scale_add_(z, 10.0)
        assert tp.allclose(z, x * 10.0)

    def test_int_list_arg(self, myext):
        x = tp.randn(2, 3)
        s = myext.sum_dims(x, [1])
        assert s.shape == (2,)
        assert tp.allclose(s, x.sum(1))

    def test_type_error_path(self, myext):
        with pytest.raises(TypeError, match="must be a Tensor"):
            myext.scale_add("not-a-tensor", 1.0)

    def test_result_is_tensorplay_tensor(self, myext):
        out = myext.scale_add(tp.ones(2), 1.0)
        assert isinstance(out, tp.Tensor)
        assert out.dtype == tp.float32
