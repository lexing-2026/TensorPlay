import ast
from pathlib import Path
import runpy
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


if __name__ == "__main__":
    unittest.main()
