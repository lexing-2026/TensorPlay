import ast
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
_audit_namespace = runpy.run_path(str(ROOT / "tools" / "audit_python_package.py"))
audit_package = _audit_namespace["audit_package"]
module_names = _audit_namespace["_module_names"]


class TestPythonPackage(unittest.TestCase):
    def test_package_layout_and_exports(self):
        self.assertEqual(audit_package(ROOT / "tensorplay"), [])

    def test_exports_do_not_use_function_locals(self):
        tree = ast.parse("def build():\n    hidden = 1\n")
        self.assertNotIn("hidden", module_names(tree))

    def test_future_controls_have_public_functions(self):
        path = ROOT / "tensorplay" / "__future__.py"
        namespace = runpy.run_path(str(path))
        self.assertEqual(namespace["get_overwrite_module_params_on_conversion"](), False)
        namespace["set_overwrite_module_params_on_conversion"](True)
        self.assertTrue(namespace["get_overwrite_module_params_on_conversion"]())
        self.assertEqual(namespace["get_swap_module_params_on_conversion"](), False)

    def test_onnx_all_contains_only_real_names(self):
        path = ROOT / "tensorplay" / "onnx" / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        all_names = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        self.assertNotIn("ExportOptions", all_names)

    def test_graph_facade_declares_core_exports(self):
        path = ROOT / "tensorplay" / "graph" / "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        all_names = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
        for name in ("CodeGen", "map_arg", "has_side_effect"):
            self.assertIn(name, imports)
            self.assertIn(name, all_names)

    def test_archive_detection_accepts_bytes(self):
        script = r"""
import importlib.util
import io
import sys
import types
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
archive_dir = root / "tensorplay" / "export" / "pt2_archive"
for name, path in {
    "tensorplay": root / "tensorplay",
    "tensorplay.export": root / "tensorplay" / "export",
    "tensorplay.export.pt2_archive": archive_dir,
}.items():
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load("tensorplay.export.pt2_archive.constants", archive_dir / "constants.py")
load(
    "tensorplay.export.pt2_archive._package_weights",
    archive_dir / "_package_weights.py",
)
archive = load("tensorplay.export.pt2_archive._package", archive_dir / "_package.py")

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w") as writer:
    writer.writestr("archive_format", "pt2")
assert archive.is_pt2_package(buffer.getvalue())
assert not archive.is_pt2_package(b"not an archive")
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(ROOT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
