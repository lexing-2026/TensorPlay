import unittest

import tensorplay as tp
import tensorplay.nn as nn


class _MultilineModule(nn.Module):
    def extra_repr(self):
        return "first\n\nthird"


class _WrapperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _MultilineModule()


class TestModuleRepr(unittest.TestCase):
    def test_parameter_containers(self):
        parameter = nn.Parameter(tp.ones(2, 3))
        parameter_list = repr(nn.ParameterList([parameter]))
        parameter_dict = repr(nn.ParameterDict({"weight": parameter}))

        self.assertEqual(
            parameter_list,
            "ParameterList(  (0): Parameter containing: "
            "[tensorplay.float32 of size 2x3])",
        )
        self.assertEqual(
            parameter_dict,
            "ParameterDict(  (weight): Parameter containing: "
            "[tensorplay.FloatTensor of size 2x3])",
        )

    def test_tensor_autograd_suffix(self):
        leaf = tp.tensor([1.0], requires_grad=True)
        value = leaf * 2

        self.assertIn(", requires_grad=True)", repr(leaf))
        self.assertIn("grad_fn=<MulScalarBackward>", repr(value))
        self.assertEqual(value.grad_fn.name, "MulScalarBackward")

    def test_nested_blank_lines_are_not_indented(self):
        self.assertEqual(
            repr(_WrapperModule()),
            "_WrapperModule(\n"
            "  (child): _MultilineModule(\n"
            "    first\n"
            "\n"
            "    third\n"
            "  )\n"
            ")",
        )


if __name__ == "__main__":
    unittest.main()
