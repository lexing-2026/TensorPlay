import tensorplay as tp
import tensorplay.nn as nn

from tensorplay.testing._internal.common_utils import TestCase, run_tests


class _MultilineModule(nn.Module):
    def extra_repr(self):
        return "first\n\nthird"


class _WrapperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _MultilineModule()


class TestModuleRepr(TestCase):
    def test_parameter_containers(self):
        parameter = nn.Parameter(tp.ones(2, 3))
        parameter_list = repr(nn.ParameterList([parameter]))
        parameter_dict = repr(nn.ParameterDict({"weight": parameter}))

        self.assertExpected(parameter_list, "parameter_list")
        self.assertExpected(parameter_dict, "parameter_dict")

    def test_tensor_autograd_suffix(self):
        leaf = tp.tensor([1.0], requires_grad=True)
        value = leaf * 2

        self.assertExpected(repr(leaf), "leaf")
        self.assertExpected(repr(value), "value")
        self.assertEqual(value.grad_fn.name, "MulScalarBackward")

    def test_nested_blank_lines_are_not_indented(self):
        self.assertExpected(repr(_WrapperModule()))


if __name__ == "__main__":
    run_tests()
