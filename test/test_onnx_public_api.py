"""Tests for the onnx public API surface (errors/verification/testing/utils)."""

import pytest

import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.onnx.errors import (
    OnnxExporterError,
    OnnxExporterWarning,
    UnsupportedOperatorError,
)
from tensorplay.onnx.testing import run_model_test
from tensorplay.onnx.utils import register_custom_op_symbolic
from tensorplay.onnx.verification import VerificationError, VerificationResult


def _small_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x)

    return Net().eval()


def test_error_hierarchy():
    assert issubclass(UnsupportedOperatorError, OnnxExporterError)
    assert issubclass(OnnxExporterError, RuntimeError)
    assert issubclass(OnnxExporterWarning, UserWarning)
    err = UnsupportedOperatorError("some_op", 18, 20)
    assert "opset version 20" in str(err)
    plain = UnsupportedOperatorError("some_op")
    assert "some_op" in str(plain)


def test_verification_result_bool():
    ok = VerificationResult(matched=True)
    bad = VerificationResult(matched=False, mismatches=["out0"])
    assert bool(ok)
    assert not bool(bad)


def test_run_model_test_matches_eager():
    model = _small_model()
    x = tp.rand(2, 4)
    outputs = run_model_test(model, x, input_names=["input"])
    reference = model(x)
    assert outputs[0].shape == tuple(reference.shape)


def test_run_model_test_requires_input_names():
    model = _small_model()
    with pytest.raises(ValueError):
        run_model_test(model, tp.rand(2, 4))


def test_register_custom_op_symbolic():
    from tensorplay.onnx._composite_ops import _ANY_MODULE_HANDLERS

    calls = []

    def symbolic(context):
        calls.append(1)
        return context.builder.op("Identity", [context.args[0].name],
                                  outputs=[context.builder.unique("identity_out")])

    register_custom_op_symbolic("my_test_op", symbolic, 18, params="")
    try:
        assert "my_test_op" in _ANY_MODULE_HANDLERS
        from tensorplay.onnx._composite_ops import lookup_function_handler

        entry = lookup_function_handler("some.module", "my_test_op")
        assert entry is not None
    finally:
        from tensorplay.onnx.utils import unregister_custom_op_symbolic

        unregister_custom_op_symbolic("my_test_op")
    assert "my_test_op" not in _ANY_MODULE_HANDLERS
